from torch.nn import AvgPool2d, BatchNorm2d, Conv2d, Linear, ReLU, Sequential

from ._container import Classifier, make_conv_pool_activ


def _make_seq(in_channels, out_channels, c_kernel_size, gc_stride, gc_kernel_size=3):
    return Sequential(
        *make_conv_pool_activ(
            in_channels,
            out_channels,
            c_kernel_size,
            bias=False,
            padding=(c_kernel_size - 1) // 2,
        ),
        BatchNorm2d(out_channels, eps=0.001),
        ReLU(),
        Conv2d(
            out_channels,
            out_channels,
            gc_kernel_size,
            bias=False,
            stride=gc_stride,
            padding=(gc_kernel_size - 1) // 2,
            groups=out_channels,
        ),
        BatchNorm2d(out_channels, eps=0.001),
        ReLU()
    )


class MobileNet(Classifier):
    def __init__(self):
        convs = Sequential(
            _make_seq(3, 32, 3, 1),
            _make_seq(32, 64, 1, 2),
            _make_seq(64, 128, 1, 1),
            _make_seq(128, 128, 1, 2),
            _make_seq(128, 256, 1, 1),
            _make_seq(256, 256, 1, 2),
            _make_seq(256, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 1),
            _make_seq(512, 512, 1, 2),
            _make_seq(512, 1024, 1, 1),
            *make_conv_pool_activ(1024, 1024, 1, padding=0, bias=False),
            BatchNorm2d(1024, eps=0.001),
            ReLU(),
            AvgPool2d(2)
        )
        linears = Sequential(Linear(1024, 10))
        super().__init__(convs, linears)
