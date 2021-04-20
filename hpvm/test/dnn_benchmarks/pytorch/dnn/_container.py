from typing import Callable, Optional

import torch
from torch.nn import Conv2d, MaxPool2d, Module, Sequential, Softmax

ActivT = Optional[Callable[[], Module]]


def make_conv_pool_activ(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    activation: ActivT = None,
    pool_size: Optional[int] = None,
    pool_stride: Optional[int] = None,
    **conv_kwargs
):
    layers = [Conv2d(in_channels, out_channels, kernel_size, **conv_kwargs)]
    if activation:
        layers.append(activation())
    if pool_size is not None:
        layers.append(MaxPool2d(pool_size, stride=pool_stride))
    return layers


class Classifier(Module):
    def __init__(
        self, convs: Sequential, linears: Sequential, use_softmax: bool = True
    ):
        super().__init__()
        self.convs = convs
        self.linears = linears
        self.softmax = Softmax(1) if use_softmax else Sequential()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.convs(inputs)
        return self.softmax(self.linears(outputs.view(outputs.shape[0], -1)))
