from typing import Iterable

from torch.nn import Linear, ReLU, Sequential

from ._container import Classifier, make_conv_pool_activ


class _VGG16(Classifier):
    def __init__(self, linear_inouts: Iterable[int]):
        convs = Sequential(
            *make_conv_pool_activ(3, 64, 3, ReLU, padding=1),
            *make_conv_pool_activ(64, 64, 3, ReLU, 2, padding=1),
            *make_conv_pool_activ(64, 128, 3, ReLU, padding=1),
            *make_conv_pool_activ(128, 128, 3, ReLU, 2, padding=1),
            *make_conv_pool_activ(128, 256, 3, ReLU, padding=1),
            *make_conv_pool_activ(256, 256, 3, ReLU, padding=1),
            *make_conv_pool_activ(256, 256, 3, ReLU, 2, padding=1),
            *make_conv_pool_activ(256, 512, 3, ReLU, padding=1),
            *make_conv_pool_activ(512, 512, 3, ReLU, padding=1),
            *make_conv_pool_activ(512, 512, 3, ReLU, 2, padding=1),
            *make_conv_pool_activ(512, 512, 3, ReLU, padding=1),
            *make_conv_pool_activ(512, 512, 3, ReLU, padding=1),
            *make_conv_pool_activ(512, 512, 3, ReLU, 2, padding=1)
        )
        linear_layers = [
            Linear(in_, out) for in_, out in zip(linear_inouts, linear_inouts[1:])
        ]
        linear_relus = [ReLU() for _ in range(2 * len(linear_layers) - 1)]
        linear_relus[::2] = linear_layers
        linears = Sequential(*linear_relus)
        super().__init__(convs, linears)


class VGG16Cifar10(_VGG16):
    def __init__(self):
        super().__init__([512, 512, 10])


class VGG16Cifar100(_VGG16):
    def __init__(self):
        super().__init__([512, 512, 100])


class VGG16ImageNet(_VGG16):
    def __init__(self):
        super().__init__([25088, 4096, 4096, 1000])
