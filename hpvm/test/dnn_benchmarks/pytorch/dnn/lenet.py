from torch.nn import Linear, Sequential, Tanh

from ._container import Classifier, make_conv_pool_activ


class LeNet(Classifier):
    def __init__(self):
        convs = Sequential(
            *make_conv_pool_activ(1, 32, 5, Tanh, 2, padding=2),
            *make_conv_pool_activ(32, 64, 5, Tanh, 2, padding=2)
        )
        linears = Sequential(Linear(7 * 7 * 64, 1024), Tanh(), Linear(1024, 10), Tanh())
        super().__init__(convs, linears)
