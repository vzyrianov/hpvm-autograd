import logging
from collections import OrderedDict
from typing import Iterable, Tuple

import pandas
import torch
import torch.nn as nn

from .indexing import ModuleIndexer

_summary_used = False
msg_logger = logging.getLogger(__name__)


def get_flops(module: nn.Module, input_shape, output_shape):
    # Partially following impl here:
    # https://github.com/juliagusak/flopco-pytorch/blob/c9679785d802f4984c9c5e5d47958e3b82044ce9/flopco/compute_layer_flops.py
    from torchvision.models.detection.transform import GeneralizedRCNNTransform

    def linear_flops():
        m, n = input_shape
        k, n_ = module.weight.shape
        assert n == n_
        return m * n * k

    def conv2d_flops():
        _, _, h, w = output_shape
        return module.weight.numel() * h * w

    def pool2d_flops():
        ksize = module.kernel_size
        if isinstance(ksize, int):
            ksize = ksize, ksize
        k_area = ksize[0] * ksize[1]
        return k_area * _get_numel(output_shape)

    def ntimes_input_numel(n: int):
        return lambda: n * _get_numel(input_shape)

    def ntimes_output_numel(n: int):
        return lambda: n * _get_numel(output_shape)

    type_dispatch = {
        nn.Linear: linear_flops,
        nn.Conv2d: conv2d_flops,
        nn.BatchNorm2d: ntimes_output_numel(6),
        nn.ReLU: ntimes_output_numel(1),
        nn.AvgPool2d: pool2d_flops,
        nn.MaxPool2d: pool2d_flops,
        # Resize is likely more than 1x input size, but let's go with that.
        GeneralizedRCNNTransform: ntimes_input_numel(2),
    }
    handler = type_dispatch.get(type(module))
    if not handler:
        if not list(module.children()):
            _warn_once(
                f"Module {module} cannot be handled; its FLOPs will be estimated as 0"
            )
        return 0.0
    try:
        return handler()
    except RuntimeError as e:
        _warn_once(
            f'Error "{e}" when handling {module}; its FLOPs will be estimated as 0'
        )
        return 0.0


def get_summary(model: nn.Module, model_args: Tuple) -> pandas.DataFrame:
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

    include = lambda m: (
        not isinstance(m, nn.Sequential)
        and not isinstance(m, nn.ModuleList)
        and not (m == model)
    )
    indexed = ModuleIndexer(model, include, lambda m: True)
    find_by_module = lambda m: indexed.find_by_module(m)[0]
    summary = OrderedDict()
    hooks = []
    special_ops = {LastLevelMaxPool: last_level_max_pool_io}

    def hook(module: nn.Module, inputs, outputs):
        module_name = find_by_module(module)
        special_handler = special_ops.get(type(module))
        if special_handler:
            input_shape, output_shape, flops = special_handler(module, inputs, outputs)
        else:
            input_shape, output_shape, flops = default_io(module, inputs, outputs)

        n_params = sum(param.numel() for param in module.parameters())
        trainable = any(param.requires_grad for param in module.parameters())
        is_leaf = not list(module.children())

        summary[module_name] = OrderedDict(
            type=module.__class__.__name__,
            input_shape=input_shape,
            output_shape=output_shape,
            params=n_params,
            flops=flops,
            trainable=trainable,
            is_leaf=is_leaf,
        )

    def register_hook(module: nn.Module):
        if include(module):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)
    with torch.no_grad():
        model(*model_args)
    # remove these hooks
    for h in hooks:
        h.remove()
    global _summary_used
    _summary_used = True  # Prevent further error printing
    return pandas.DataFrame(summary).T


def last_level_max_pool_io(_, inputs, outputs):
    input_shapes = [list(i.size()) for i in inputs[0]]
    output_shapes = [list(o.size()) for o in outputs[0]]
    total_numel = sum([_get_numel(s) for s in input_shapes])
    return input_shapes, output_shapes, total_numel


def default_handle_sizes(value):
    try:
        if isinstance(value, torch.Tensor):
            return list(value.size())
        if isinstance(value, dict):
            return {k: list(v.size()) for k, v in value.items()}
        if isinstance(value, Iterable):
            return [list(i.size()) for i in value]
    except AttributeError as e:
        _warn_once(f"Cannot get shape of {type(value)}: error {e}")
        return None
    _warn_once(f"Cannot get shape of {type(value)}")
    return None


def default_io(module: nn.Module, inputs, outputs):
    input_shape = default_handle_sizes(inputs[0])
    output_shape = default_handle_sizes(outputs)
    return input_shape, output_shape, get_flops(module, input_shape, output_shape)


def _get_numel(shape):
    return torch.prod(torch.tensor(shape)).item()


def _warn_once(*args, **kwargs):
    if _summary_used:
        return
    msg_logger.warning(*args, **kwargs)
