"""Approximation techniques for torch.nn layers."""
from pathlib import Path
from typing import Dict, Iterable, Set, Type

import torch
from torch.nn import Conv2d, Linear, Module, Parameter

from .._logging import PathLike
from ..torchapp import TorchApproxKnob
from ._copy import module_only_deepcopy


def _interpolate_first_dim(tensor: torch.Tensor, interp_indices: Iterable[int]):
    def tensor_at(idx_: int):
        if idx_ in interp_indices:
            raise IndexError
        if idx_ < 0 or idx_ >= tensor.size()[0]:
            return torch.zeros_like(tensor[0])
        return tensor[idx_]

    for idx in interp_indices:
        if idx < 0 or idx >= tensor.size()[0]:
            raise IndexError
        elif idx == 0:  # First row
            tensor[idx] = tensor_at(1)
        elif idx == tensor.size()[0] - 1:  # Last row
            tensor[idx] = tensor_at(idx - 1)
        else:  # Middle rows
            tensor[idx] = (tensor_at(idx - 1) + tensor_at(idx + 1)) / 2.0
    return tensor


class PerforateConv2dStride(TorchApproxKnob):
    r"""Simulation of strided perforated convolution for `torch.nn.Conv2d`.

    Perforated convolution skips computing some entries in the output and instead interpolates
    these values, to reduce the number of float-ops needed to complete a convolution op.
    In this implementation, selected rows or columns of the output are discarded and replaced
    with linearly interpolated values from the neighboring rows or columns. Each channel is
    considered independently.
    This implementation gives the same output as actual perforated convolution but without the
    performance benefit.

    Parameters
    ----------
    direction_is_row : bool
        If True, discard and interpolate rows, otherwise columns.
    stride : int \in [2, +\infty)
        Skip 1 row/column in the convolution kernel per `stride` elements.
    offset : int \in [0, stride)
        Skipped first row/column is `offset`.

    Attributes
    ----------
    interp_axis : int :math:`\in \{2, 3\}`
        The axis that will be perforated over. As the input is an NCHW tensor, if
        `direction_is_row` then `interp_axis = 2`, otherwise `interp_axis = 3`.
    stride : int :math:`\in [2, +\infty)`
        Equal to parameter `stride`.
    offset : int :math:`\in [0, stride)`
        Equal to parameter `offset`.
    """

    def __init__(
        self,
        name: str,
        direction_is_row: bool,
        stride: int,
        offset: int,
        use_fp16: bool,
        exp_speedup: float,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        assert stride >= 2
        assert 0 <= offset < stride
        self.interp_axis = 2 if direction_is_row else 3
        self.stride = stride
        self.offset = offset
        self.fp16 = use_fp16
        self.exp_speedup = exp_speedup

    def is_applicable(self, op: Module) -> bool:
        return isinstance(op, Conv2d)

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def expected_speedup(self) -> float:
        return self.exp_speedup

    class PerforateConv2dStrideModule(Module):
        def __init__(self, conv: Conv2d, approx: "PerforateConv2dStride"):
            super().__init__()
            self.conv = conv
            self.approx = approx
            if self.approx.fp16:
                self.conv = self.conv.half()

        def conv_no_bias(self, x: torch.Tensor):
            if self.conv.bias is None:
                return self.conv(x)
            bias = self.conv.bias
            self.conv.bias = None
            result = self.conv(x)
            self.conv.bias = bias
            return result

        def add_conv_bias(self, conv_output: torch.Tensor):
            if self.conv.bias is None:
                return conv_output
            broadcast_bias = self.conv.bias.reshape(1, -1, 1, 1)
            return conv_output + broadcast_bias

        def forward(self, x: torch.Tensor):
            if self.approx.fp16:
                x = x.half()
            x = self.conv_no_bias(x)
            assert x.dim() == 4
            # Put self.approx.interp_axis to first axis temporarily
            x = x.transpose(0, self.approx.interp_axis)
            interp_indices = torch.tensor(
                range(self.approx.offset, x.size(0), self.approx.stride)
            )
            x = _interpolate_first_dim(x, interp_indices)
            # Putting axes back
            x = x.transpose(0, self.approx.interp_axis)
            x = self.add_conv_bias(x)
            if self.approx.fp16:
                assert x.dtype == torch.float16
            return x.float()

    def apply(self, module: Conv2d) -> PerforateConv2dStrideModule:
        return self.PerforateConv2dStrideModule(module, self)


class Conv2dSampling(TorchApproxKnob):
    r"""Simulation of sampled convolution for `torch.nn.Conv2d`.

    Skips some elements of the convolution kernel in a uniform, strided manner,
    to reduce the amount of float-ops needed to compute each output entry.
    This implementation gives the same output as actual sampled convolution but without the
    performance benefit.

    Parameters
    ----------
    skip_every: int
        Skip 1 element in the convolution kernel per `skip_every` elements.
    skip_offset : int :math:`\in [0, +\infty)`
        Index of first element to be skipped.
        For example, if `skip_every = 3` and `skip_offset = 1`, then indices skipped
        will be [1, 4, 7, ...]
    interp_rate : float
        The weight will be compensated ("interpolated") with a ratio after skipping elements,
        which is naturally equal to :math:`1 + (1 / (skip\_every - 1)`.
        `interp_rate` modifies this rate to :math:`1 + (1 / (skip\_every - 1) \times interp\_rate`.
    use_fp16 : bool
        Whether to use fp16 weight/input or not.
    """

    def __init__(
        self,
        name: str,
        skip_every: int,
        skip_offset: int,
        interp_rate: float,
        use_fp16: bool,
        exp_speedup: float,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        assert skip_every >= 2 and skip_offset >= 0
        self.skip_every = skip_every
        self.skip_offset = skip_offset
        self.interp_rate = interp_rate
        self.fp16 = use_fp16
        self.exp_speedup = exp_speedup

    def is_applicable(self, op: Module) -> bool:
        return isinstance(op, Conv2d)

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def expected_speedup(self) -> float:
        return self.exp_speedup

    @staticmethod
    def sample_conv_weight(
        interp_rate: float, skip_every: int, skip_offset: int, weight: torch.Tensor
    ):
        r"""Samples (skips & interpolates) convolution kernel according to parameters.

        For a given `weight` tensor of shape `(C1, C2, H, W)`, sample each output channel
        (on axis 0) independently.
        Flatten each output channel tensor into 1 dim.
        In normal cases, set elements at indices ``range(skip_offset, C_2 * H * W, skip_every)``
        to 0.
        However, if `skip_every` == `h` == `w` == 3, we may end up skipping the same whole rows for
        each input channel, which is undesirable.
        Instead, increment the offset by 1 for each input channel.
        Last, multiplies the kernel by the inverse ratio of elements dropped for an interpolation.
        """
        if len(weight.shape) != 4:
            raise ValueError("Conv2d weight should be 4-dimensional")
        c1, c2, h, w = weight.shape
        if skip_every == h == w == 3:
            # Indices (0..h*w) to skip for each input channel
            per_chan_skip_indices = [
                range((i_chan + skip_offset) % skip_every, h * w, skip_every)
                for i_chan in range(c2)
            ]
            # Indices (0..c2*h*w) for each output channel, created by adding i*h*w for ith channel.
            skip_indices = torch.tensor(
                [
                    x + i * h * w
                    for i, per_chan in enumerate(per_chan_skip_indices)
                    for x in per_chan
                ]
            )
        else:
            # Indices (0..c2*h*w) to skip for each output channel
            skip_indices = torch.arange(skip_offset, c2 * h * w, skip_every)
        flat_weight = weight.reshape(c1, -1).clone()
        flat_weight[:, skip_indices] = 0
        interp_rate = 1 + (1 / (skip_every - 1) * interp_rate)
        flat_weight *= interp_rate
        return flat_weight.reshape_as(weight)

    def apply(self, module: Conv2d) -> Conv2d:
        # Only copy the submodules, weights are still shared.
        copied = module_only_deepcopy(module)
        # But only write over the weight of copied version, original weight is unchanged.
        copied.weight = Parameter(
            self.sample_conv_weight(
                self.interp_rate, self.skip_every, self.skip_offset, copied.weight
            )
        )
        return copied


def _quantize_uint8(
    tensor: torch.Tensor, range_min: float, range_max: float
) -> torch.Tensor:
    """Simulates quantization of `tensor` down to uint8, while still returning float values.

    In the returned tensor the data will NOT be in [0, 255] range, but only 256 unique float
    value will exist.
    """

    quantize_range = 256
    input_range = range_max - range_min
    mul = input_range / quantize_range
    # Map tensor into [0, 256] range.
    affined = (tensor - range_min) / mul
    # Convert tensor to int and back to float so it will have
    # 256 (actually 257!; following hpvm impl) unique float values [0, 256].
    # Then reverse affine it to the original range.
    quanted = torch.floor(affined).to(torch.int).to(torch.float)
    quanted_float = quanted * mul + range_min
    # Clip tensor
    return torch.clamp(quanted_float, range_min, range_max)


class PromiseSim(TorchApproxKnob):
    """Simulates analog accelerator PROMISE.

    This hardware is proposed in "PROMISE: An End-to-End Design of a Programmable Mixed-Signal
    Accelerator for Machine-Learning Algorithms."
    """

    scaling_values = [0.75, 0.64, 0.336, 0.21, 0.168, 0.14, 0.11, 0.0784, 0.005]

    def __init__(self, name: str, noise_level: int, exp_speedup: float, **kwargs):
        super().__init__(name, **kwargs)
        self.noise_level = noise_level
        self.exp_speedup = exp_speedup

    def is_applicable(self, op: Module) -> bool:
        return isinstance(op, (Conv2d, Linear))

    @property
    def deterministic(self) -> bool:
        return False

    @property
    def expected_speedup(self) -> float:
        return self.exp_speedup

    def add_promise_noise(self, tensor: torch.Tensor):
        scale = self.scaling_values[self.noise_level]
        noise = torch.normal(
            mean=0.0, std=scale, size=tensor.size(), device=tensor.device
        )
        return noise * tensor + tensor

    class PromiseSimModule(Module):
        def __init__(self, module: Conv2d, approx: "PromiseSim"):
            super().__init__()
            if not hasattr(module, "conv_ranges"):
                raise ValueError(
                    f"Quantization range of conv2d layer {module} not found"
                )
            self.input_r, weight_r, bias_r, self.output_r = module.conv_ranges
            module.weight.data = _quantize_uint8(module.weight, *weight_r)
            if module.bias is not None:
                module.bias.data = _quantize_uint8(module.bias, *bias_r)
            self.module = module
            self.approx = approx

        def forward(self, input_: torch.Tensor) -> torch.Tensor:
            # Quantize input, weight, bias (see __init__), and add noise to input.
            input_ = _quantize_uint8(input_, *self.input_r)
            input_ = self.approx.add_promise_noise(input_)
            output = self.module(input_)
            # Then again, quantize output.
            return _quantize_uint8(output, *self.output_r)

    def apply(self, module: Conv2d, **kwargs) -> PromiseSimModule:
        return self.PromiseSimModule(module, self)


class FP16Approx(TorchApproxKnob):
    """
    Approximates by reducing precision of layer computation to float16.

    This is the baseline knob for GPU device by default.
    """

    def __init__(self, name: str, exp_speedup: float, **kwargs):
        super().__init__(name, **kwargs)
        self.exp_speedup = exp_speedup

    @property
    def deterministic(self) -> bool:
        return True

    @property
    def expected_speedup(self) -> float:
        return self.exp_speedup

    def is_applicable(self, op: Module) -> bool:
        return True

    class FP16ApproxModule(Module):
        def __init__(self, module: Module):
            super().__init__()
            self.module = module.half()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.module(x.half())
            assert x.dtype == torch.float16
            return x.float()

    def apply(self, module: Module) -> FP16ApproxModule:
        return self.FP16ApproxModule(module)


class FP32Baseline(TorchApproxKnob):
    @property
    def deterministic(self) -> bool:
        return True

    @property
    def expected_speedup(self) -> float:
        return 1.0

    def is_applicable(self, op: Module) -> bool:
        return True

    def apply(self, op: Module) -> Module:
        return op


default_name_to_class = {
    k.__name__: k
    for k in [
        FP32Baseline,
        FP16Approx,
        PromiseSim,
        PerforateConv2dStride,
        Conv2dSampling,
    ]
}
default_knob_file = Path(__file__).parent / "default_approx_params.json"


def get_knobs_from_file(
    filepath: PathLike = default_knob_file,
    extra_name_to_class: Dict[str, Type[TorchApproxKnob]] = None,
) -> Set[TorchApproxKnob]:
    """get_knobs_from_file(filepath=default_knob_file, extra_name_to_class=None)

    Constructs and returns a set of `TorchApproxKnob` from a knob declaration file.
    `default_knob_file` points to a file that is contained in the predtuner package,
    so just calling ``get_knobs_from_file()`` should provide a set of predefined knobs already.

    :param filepath: the knob declaration file (JSON) to read from.
    :param extra_name_to_class: a mapping from the name of the approximation to the
           class (implementation) of the approximation.
           If not given, only the builtin approximations will be considered
           when parsing the declaration file.
    :type extra_name_to_class: Dict[str, Type[TorchApproxKnob]]
    :rtype: Set[TorchApproxKnob]
    """

    import json

    extra_name_to_class = extra_name_to_class or {}
    default_names = set(list(default_name_to_class))
    extra_names = set(list(extra_name_to_class))
    if default_names.intersection(extra_names):
        raise ValueError(
            f"Provided extra class names clash with default class names; \n"
            f"Default: {default_names}\n"
            f"Extra: {extra_names}"
        )
    name_to_class = {**default_name_to_class, **extra_name_to_class}
    filepath = Path(filepath)
    with filepath.open() as f:
        knobs_json = json.load(f)
    ret = set()
    for knob_dict in knobs_json:
        if not isinstance(knob_dict, dict):
            raise ValueError(f"JSON file for knob initialization contains non-dict")
        # 'class' is not a valid argument name in Python, so we can use this key.
        class_name = knob_dict.pop("class")
        if class_name not in name_to_class:
            raise KeyError(f"{class_name} not found among knob class names")
        class_ty = name_to_class[class_name]
        try:
            ret.add(class_ty(**knob_dict))
        except TypeError as e:
            raise TypeError(
                f"Approximation class {class_name} does not accept given arguments {knob_dict}.\n"
                f"Original exception: {e}"
            )
    return ret
