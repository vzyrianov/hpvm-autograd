import abc
from os import PathLike
from typing import List, Optional, Sequence, Tuple

import numpy as np
import onnx

ShapeT = Optional[List[int]]


class DFGNode(abc.ABC):
    """Abstract node that represents 1 instruction in HPVM.
    
    op_type should be overriden in subclasses for readability.
    """

    op_type = ""
    hpvm_op_type = ""

    def __init__(
        self, name: str, input_shapes: Sequence[ShapeT], output_shape: ShapeT, **kwargs
    ):
        self.name = name
        self.input_shapes = input_shapes
        self.output_shape = output_shape
        self.attrs = kwargs

    def codegen(self) -> Tuple[str, list]:
        return "", []

    def hpvm_codegen(self) -> Tuple[str, list]:
        return "", []

    def get_flops(self) -> int:
        return np.prod(self.output_shape) if self.output_shape else 0

    def __repr__(self) -> str:
        sin = " x ".join(str(sh) if sh else "??" for sh in self.input_shapes)
        sout = self.output_shape if self.output_shape else "??"
        if sin:
            return f"{self.name}({self.op_type}): {sin} -> {sout} ({self.get_flops()})"
        else:
            return f"{self.name}({self.op_type}): {sout}"


class TensorNode(DFGNode, abc.ABC):
    """An abstract node for a value that exists without an instruction.

    This is akin to Value class in LLVM, but in a different place on the
    inheritance tree."""

    def __init__(self, proto: onnx.TensorProto, shape: ShapeT, new_name: str):
        if not proto.name.strip():
            raise ValueError("Tensor's name is required.")
        super().__init__(proto.name, [], shape)
        self.new_name = new_name

    def __str__(self):
        return f"{self.__class__.__name__}: {self.name}"

    __repr__ = __str__


class InputTensor(TensorNode):
    """Input to the computation graph.
    
    This is basically only used for its information about the ONNX input,
    itself doesn't emit instruction or any interesting thing.
    """

    op_type = "InputTensor"

    def __init__(self, input_proto: onnx.TensorProto, shape: ShapeT, new_name: str):
        super().__init__(input_proto, shape, new_name)
        # get type of input tensor
        tensor_type = input_proto.type.tensor_type
        # check if it has a shape:
        shape = tensor_type.shape
        self.shape: List[int] = [d.dim_value for d in shape.dim]


class WeightTensor(TensorNode):
    """An initialized parameter in ONNX graph.
    
    This is any parameter that has a initializer value in the ONNX model
    (as opposed to InputTensor, which doesn't have any value).
    """

    op_type = "WeightTensor"

    def __init__(self, weight_proto: onnx.TensorProto, new_name: str):
        from onnx import numpy_helper

        self.input_data = numpy_helper.to_array(weight_proto)
        sh = self.input_data.shape
        if len(sh) == 1:
            shape = [1, sh[0], 1, 1]
        elif len(sh) == 2:
            shape = [1, 1, sh[0], sh[1]]
        elif len(sh) == 4:
            shape = [sh[0], sh[1], sh[2], sh[3]]
        else:
            shape = [1] * 4
        super().__init__(weight_proto, shape, new_name)

    def dump_weight(self, file_name: PathLike):
        self.input_data.tofile(file_name)

    def transpose_(self):
        if len(self.input_data.shape) != 2:
            raise ValueError("Can only transpose 2D array")
        self.input_data = self.input_data.T
        self.output_shape[3], self.output_shape[2] = self.output_shape[2:]


class Conv2DNode(DFGNode):
    op_type = "Conv2D"

    def __init__(
        self,
        name: str,
        input_shapes: Tuple[ShapeT, ShapeT],
        output_shape: ShapeT,
        pads: Sequence[int],
        strides: Sequence[int],
        dilations: Sequence[int],
        group: int,
        kernel_shape,
    ):
        super().__init__(name, input_shapes, output_shape)
        assert len(pads) == 4, "2D convolution must have 4 padding values"
        if any(p != pads[0] for p in pads[1:]):
            raise ValueError("Convolution with different padding is unsupported")
        if list(dilations) != [1, 1]:
            raise ValueError("Dilation > 1 is unsupported")
        self.group = group
        if group == 1:
            self.hpvm_op_type = "convolution"
        else:
            # HPVM tensor_runtime distinguishes these two. (sigh)
            self.hpvm_op_type = "depthwise_convolution"
        self.pads = pads[0]
        self.sh, self.sw = strides

    def codegen(self):
        return (
            "tensorConvolution",
            [self.pads, self.pads, self.sh, self.sw, 1, self.group],
        )

    def hpvm_codegen(self):
        if self.group != 1:
            return (
                "__hpvm__tensor_group_convolution",
                # 1 is conv_mode -- should always be 1
                [self.pads, self.pads, self.sh, self.sw, 1, self.group],
            )
        return (
            "__hpvm__tensor_convolution",
            [self.pads, self.pads, self.sh, self.sw],
        )

    def get_flops(self) -> int:
        _, kshape = self.input_shapes
        if not self.output_shape or not kshape:
            return 0
        _, _, h, w = self.output_shape
        c1, c2, kh, kw = kshape
        return int(c1 * c2 * h * w * kh * kw / (self.sh * self.sw))


class _Pool2DNode(DFGNode, abc.ABC):
    """Common super class of Average pooling and Max pooling."""

    pool_type = "0"

    def __init__(
        self,
        name: str,
        input_shapes: Tuple[ShapeT, ShapeT],
        output_shape: ShapeT,
        strides: Sequence[int],
        kernel_shape: Sequence[int],
        pads: Sequence[int],
        ceil_mode: int = 0,
    ):
        super().__init__(name, input_shapes, output_shape)
        self.strides = sh, sw = strides
        self.kernel_shape = kernel_shape
        pt, pb, pl, pr = pads
        if pt != pb or pl != pr:
            raise ValueError(
                "Unequal padding on either side of same axis is unsupported"
            )
        self.pads = pt, pl
        if ceil_mode != 0:
            raise ValueError("Only ceil_mode == 0 is supported")

    def codegen(self):
        return (
            "tensorPooling",
            [self.pool_type, *self.kernel_shape, *self.pads, *self.strides,],
        )

    def get_flops(self) -> int:
        input0 = self.input_shapes[0]
        return np.prod(input0) if input0 else 0


class MaxPool2DNode(_Pool2DNode):
    pool_type = "0"
    op_type = "MaxPool2D"
    hpvm_op_type = "maxpool"

    def hpvm_codegen(self):
        return (
            "__hpvm__tensor_pool_max",
            [*self.kernel_shape, *self.pads, *self.strides],
        )


class AveragePool2DNode(_Pool2DNode):
    pool_type = "1"
    op_type = "AveragePool2D"
    hpvm_op_type = "avgpool"

    def hpvm_codegen(self):
        return (
            "__hpvm__tensor_pool_mean",
            [*self.kernel_shape, *self.pads, *self.strides],
        )


class BiasAddNode(DFGNode):
    op_type = "BiasAdd"
    hpvm_op_type = "add"

    def codegen(self):
        return "tensorAdd", []

    def hpvm_codegen(self):
        return "__hpvm__tensor_add", []


class MatMulNode(DFGNode):
    op_type = "MatMul"
    hpvm_op_type = "linear"

    def codegen(self):
        return "tensorGemmGPU", []

    def hpvm_codegen(self):
        return "__hpvm__tensor_mul", []

    def gemm_transpose(self, predec):
        """Find and transpose weights of the onnx gemm node.
        
        This way we transpose the constant weight instead of exporting
        a transpose node (which doesn't yet exist in HPVM).
        """

        def _transpose(idx: int):
            weight = predec[idx]
            if not isinstance(weight, WeightTensor):
                raise ValueError(
                    f"Cannot transpose non-const {weight} (transpose op needed)"
                )
            weight.transpose_()
            self.input_shapes[idx] = weight.output_shape

        # Some tensors may need transposing
        if self.attrs.get("transA", False):
            _transpose(0)
        if self.attrs.get("transB", False):
            _transpose(1)

    def get_flops(self) -> int:
        ishape, wshape = self.input_shapes
        if not ishape or not wshape:
            return 0
        input_len = np.prod(ishape)
        _, _, len_, k = wshape
        assert input_len == len_
        return input_len * k


class SoftMaxNode(DFGNode):
    op_type = "SoftMax"
    hpvm_op_type = "softmax"

    def codegen(self):
        return "tensorSoftmax", []

    def hpvm_codegen(self):
        return "__hpvm__tensor_softmax", []


class AddNode(DFGNode):
    op_type = "Add"
    hpvm_op_type = "add"

    def codegen(self):
        return "tensorAdd", []

    def hpvm_codegen(self):
        return "__hpvm__tensor_add", []


class ReluNode(DFGNode):
    op_type = "ReLU"
    hpvm_op_type = "relu"

    def codegen(self):
        return "tensorRelu", []

    def hpvm_codegen(self):
        return "__hpvm__tensor_relu", []


class TanhNode(DFGNode):
    op_type = "Tanh"
    hpvm_op_type = "tanh"

    def codegen(self):
        return "tensorTanh", []

    def hpvm_codegen(self):
        return "__hpvm__tensor_tanh", []


class BatchNormalizationNode(DFGNode):
    op_type = "BatchNorm"
    hpvm_op_type = "batchnorm"

    def __init__(
        self,
        name: str,
        input_shapes: Tuple[ShapeT, ShapeT],
        output_shape: ShapeT,
        epsilon: float,
        axis: int = None,
        momentum: float = None,
    ):
        super().__init__(name, input_shapes, output_shape)
        self.epsilon = epsilon

    def codegen(self):
        return "tensorBatchNorm", [self.epsilon]

    def hpvm_codegen(self):
        return "__hpvm__tensor_batchnorm", [self.epsilon]


class FlattenNode(DFGNode):
    op_type = "Flatten"


class ActivationNode(DFGNode):
    """
    Element wise operators that is for activation function
    e.g. HardSigmoid, LeakyRelu, PRelu, Pow, Reciprocal,
    Relu, Selu, Sigmoid, Softplus, Sqrt, ThresholdedRelu,
    Abs, Ceil, Elu, Floor, Neg
    """

    pass


class LogicalOpNode(DFGNode):
    """
    Element wise operators that is not for activation function.
    In other words, they are logical comparison operators
    e.g. And, Equal, Greater, GreaterOrEqual, Less, LessOrEqual,
    Or, Xor
    """

    pass


class ZeroPadding2DNode(DFGNode):
    pass


class DepthwiseConv2DNode(DFGNode):
    pass


class DenseNode(DFGNode):
    pass


class PadNode(DFGNode):
    pass


class IdentityNode(DFGNode):
    pass
