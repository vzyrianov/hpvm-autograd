from torch.nn import Linear, ReLU, Sequential, Tanh

from ._container import Classifier, make_conv_pool_activ


class AlexNet(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(3, 64, 11, Tanh, pool_size=2, padding=5),
            *make_conv_pool_activ(64, 192, 5, Tanh, pool_size=2, padding=2),
            *make_conv_pool_activ(192, 384, 3, Tanh, padding=1),
            *make_conv_pool_activ(384, 256, 3, Tanh, padding=1),
            *make_conv_pool_activ(256, 256, 3, Tanh, pool_size=2, padding=1)
        )
        linears = Sequential(Linear(4096, 10))
        super().__init__(convs, linears)


class AlexNet2(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(3, 32, 3, Tanh, padding=1),
            *make_conv_pool_activ(32, 32, 3, Tanh, pool_size=2, padding=1),
            *make_conv_pool_activ(32, 64, 3, Tanh, padding=1),
            *make_conv_pool_activ(64, 64, 3, Tanh, pool_size=2, padding=1),
            *make_conv_pool_activ(64, 128, 3, Tanh, padding=1),
            *make_conv_pool_activ(128, 128, 3, Tanh, pool_size=2, padding=1)
        )
        linears = Sequential(Linear(2048, 10))
        super().__init__(convs, linears)


class AlexNetImageNet(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(
                3, 64, 11, ReLU, padding=2, stride=4, pool_size=3, pool_stride=2
            ),
            *make_conv_pool_activ(
                64, 192, 5, ReLU, padding=2, pool_size=3, pool_stride=2
            ),
            *make_conv_pool_activ(192, 384, 3, ReLU, padding=1),
            *make_conv_pool_activ(384, 256, 3, ReLU, padding=1),
            *make_conv_pool_activ(
                256, 256, 3, ReLU, padding=1, pool_size=3, pool_stride=2
            )
        )
        linears = Sequential(
            Linear(9216, 4096), ReLU(), Linear(4096, 4096), ReLU(), Linear(4096, 1000),
        )
        super().__init__(convs, linears)
