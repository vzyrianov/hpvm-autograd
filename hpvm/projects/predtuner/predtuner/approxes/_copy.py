import copy
from typing import TypeVar

from torch.nn import Module, Parameter

T = TypeVar("T")


def module_only_deepcopy(obj: T, memo=None) -> T:
    """Recursively copy but only modules, not the weights.
    In the return value, all weights are still shared with those in `obj`."""

    memo = {}

    def recursive_scan_parameters(obj_):
        # Don't recurse down primitive types
        if isinstance(obj_, (int, float, bool, complex, str, bytes)):
            return
        # Additionally share all buffers of Module. For example, this accounts for
        # running_{mean|var} in BatchNorm.
        if isinstance(obj_, Module):
            buffers = obj_.__dict__.get("_buffers")
            for buffer in buffers.values():
                memo[id(buffer)] = buffer
        # Share all parameters.
        if isinstance(obj_, Parameter):
            memo[id(obj_)] = obj_
        # Walk down all other types.
        elif isinstance(obj_, dict):
            for k in obj_.keys():
                recursive_scan_parameters(k)
            for v in obj_.values():
                recursive_scan_parameters(v)
        elif isinstance(obj_, (list, tuple)):
            for x in obj_:
                recursive_scan_parameters(x)
        elif hasattr(obj_, "__dict__"):
            for x in obj_.__dict__.values():
                recursive_scan_parameters(x)

    # Populate `memo`, and then deepcopy with `memo` so that things in memo are not copied.
    recursive_scan_parameters(obj)
    # noinspection PyArgumentList
    copied = copy.deepcopy(obj, memo)
    return copied
