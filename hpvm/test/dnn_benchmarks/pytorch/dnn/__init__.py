from pathlib import Path
from typing import Union

import torch
from torch2hpvm import BinDataset, ModelExporter

from .alexnet import AlexNet, AlexNet2, AlexNetImageNet
from .datasets import CIFAR, MNIST, ImageNet
from .lenet import LeNet
from .mobilenet import MobileNet
from .resnet import ResNet18, ResNet50
from .vgg16 import VGG16Cifar10, VGG16Cifar100, VGG16ImageNet

# DNN name -> (DNN class, input_channel, input_size, suggested_batchsize)
benchmarks = {
    "lenet_mnist": (LeNet, 1, 28, 1000),
    "alexnet_cifar10": (AlexNet, 3, 32, 500),
    "alexnet2_cifar10": (AlexNet2, 3, 32, 500),
    "alexnet_imagenet": (AlexNetImageNet, 3, 224, 500),
    "mobilenet_cifar10": (MobileNet, 3, 32, 500),
    "resnet18_cifar10": (ResNet18, 3, 32, 500),
    "resnet50_imagenet": (ResNet50, 3, 224, 25),
    "vgg16_cifar10": (VGG16Cifar10, 3, 32, 500),
    "vgg16_cifar100": (VGG16Cifar100, 3, 32, 500),
    "vgg16_imagenet": (VGG16ImageNet, 3, 224, 10),
}


def export_example_dnn(
    dnn_name: str, output_dir: Union[Path, str], generate_for_tuning: bool
):
    self_folder = Path(__file__).parent.absolute()
    dnn_bench_dir = self_folder / "../.."

    model_cls, nch, img_size, batch_size = benchmarks[dnn_name]
    dataset_shape = 5000, nch, img_size, img_size
    params = dnn_bench_dir / "model_params" / dnn_name
    bin_tuneset = BinDataset(
        params / "tune_input.bin", params / "tune_labels.bin", dataset_shape
    )
    bin_testset = BinDataset(
        params / "test_input.bin", params / "test_labels.bin", dataset_shape
    )
    model: Module = model_cls()
    checkpoint = dnn_bench_dir / f"model_params/pytorch/{dnn_name}.pth.tar"
    model.load_state_dict(torch.load(checkpoint.as_posix()))

    build_dir = output_dir / "build"
    target_binary = build_dir / dnn_name
    if generate_for_tuning:
        exporter = ModelExporter(
            model, bin_tuneset, bin_testset, output_dir, target="hpvm_tensor_inspect"
        )
    else:
        conf_file = (
            dnn_bench_dir / "hpvm-c/benchmarks" / dnn_name / "data/tuner_confs.txt"
        ).absolute()
        exporter = ModelExporter(
            model, bin_tuneset, bin_testset, output_dir, config_file=conf_file
        )
    exporter.generate(batch_size=batch_size).compile(target_binary, build_dir)
    return target_binary, exporter
