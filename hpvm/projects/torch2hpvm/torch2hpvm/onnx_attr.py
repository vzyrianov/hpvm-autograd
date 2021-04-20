from typing import Dict, List, Optional, Tuple

import numpy as np
from onnx import AttributeProto, NodeProto, TensorProto, GraphProto, TensorShapeProto


def throw_ctor(ty):
    def _throw_ctor(x):
        raise ValueError(f"Cannot construct type {ty} from value {x}")

    return _throw_ctor


def composite_ctor(singular_ctor):
    def _composite_ctor(xs):
        return [singular_ctor(x) for x in xs]

    return _composite_ctor


def tensor_ctor(x: TensorProto) -> np.ndarray:
    import numpy as np

    tensor_typenames_to_numpy_ty = {
        "BOOL": np.bool,
        "UINT8": np.uint8,
        "INT8": np.int8,
        "UINT16": np.uint16,
        "INT16": np.int16,
        "INT32": np.int32,
        "UINT32": np.uint32,
        "INT64": np.int64,
        "UINT64": np.uint64,
        "STRING": np.str,
        "FLOAT16": np.float16,
        "FLOAT": np.float32,
        "DOUBLE": np.float64,
        # 'BFLOAT16' -- unsupported
        "COMPLEX64": np.complex64,
        "COMPLEX128": np.complex128,
    }

    get_tensor_typename = TensorProto.DataType.Name
    tensor_typename = get_tensor_typename(x.data_type)
    if tensor_typename not in tensor_typenames_to_numpy_ty:
        raise ValueError(f"Tensor with type {tensor_typename} cannot be processed")
    numpy_dtype = tensor_typenames_to_numpy_ty[tensor_typename]
    numpy_arr = np.frombuffer(x.raw_data, dtype=numpy_dtype).reshape(x.dims).copy()
    return numpy_arr


def parse_node_attr(onnx_attr: AttributeProto) -> Tuple[str, object]:
    from collections import namedtuple

    AttrMeta = namedtuple("AttrMeta", ["ctor", "data_name"])
    attr_typenames_to_meta = {
        "FLOAT": AttrMeta(float, "f"),
        "INT": AttrMeta(int, "i"),
        "STRING": AttrMeta(str, "s"),
        "TENSOR": AttrMeta(tensor_ctor, "t"),
        "GRAPH": AttrMeta(throw_ctor("GRAPH"), "g"),
        "SPARSE_TENSOR": AttrMeta(throw_ctor("SPARSE_TENSOR"), "sparse_tensor"),
        "FLOATS": AttrMeta(composite_ctor(float), "floats"),
        "INTS": AttrMeta(composite_ctor(int), "ints"),
        "STRINGS": AttrMeta(composite_ctor(str), "strings"),
        "TENSORS": AttrMeta(composite_ctor(tensor_ctor), "tensors"),
        "GRAPHS": AttrMeta(throw_ctor("GRAPHS"), "graphs"),
        "SPARSE_TENSORS": AttrMeta(throw_ctor("SPARSE_TENSORS"), "sparse_tensors"),
    }

    get_attr_typename = AttributeProto.AttributeType.Name
    typename = get_attr_typename(onnx_attr.type)
    assert (
        typename in attr_typenames_to_meta
    ), f"ONNX attribute contains non-ONNX type {typename}"
    attr_meta = attr_typenames_to_meta[typename]
    data = getattr(onnx_attr, attr_meta.data_name)
    parsed_data = attr_meta.ctor(data)
    return parsed_data


def node_attr_to_dict(onnx_node: NodeProto):
    return {attr.name: parse_node_attr(attr) for attr in onnx_node.attribute}


def get_node_shape(node: NodeProto) -> List[int]:
    return [dim.dim_value for dim in node.type.tensor_type.shape.dim]


def node_to_shape(onnx_graph: GraphProto) -> Dict[NodeProto, Optional[List[int]]]:
    def unique_output_name(node: NodeProto) -> str:
        if len(node.output) != 1:
            raise ValueError(f"Node {node} has more than 1 outputs")
        return node.output[0]

    out_name_to_shape = {vi.name: get_node_shape(vi) for vi in onnx_graph.value_info}
    # Add model's output shape into out_name_to_shape
    if len(onnx_graph.output) != 1:
        raise ValueError("Model doesn't have unique output")
    model_output = onnx_graph.output[0]
    out_name_to_shape[model_output.name] = get_node_shape(model_output)
    return {n: out_name_to_shape.get(unique_output_name(n)) for n in onnx_graph.node}
