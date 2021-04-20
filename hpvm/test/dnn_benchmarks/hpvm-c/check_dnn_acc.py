#!/usr/bin/env python3
from sys import argv

network_accuracies = {
    "alexnet2_cifar10": 84.98,
    "alexnet_cifar10": 79.28,
    "alexnet_imagenet": 56.30,
    "lenet_mnist": 98.70,
    "mobilenet_cifar10": 84.42,
    "resnet18_cifar10": 89.56,
    "resnet50_imagenet": 75.10,
    "vgg16_cifar10": 89.96,
    "vgg16_cifar100": 66.50,
    "vgg16_imagenet": 69.46,
}


def almost_equal(x1, x2):
    return abs(x1 - x2) < 5e-2


_, acc_file, network_name = argv
# cudnn version should have the same accuracy as non-cudnn version.
network_name = network_name.replace("_cudnn", "")
with open(acc_file) as f:
    obtained_acc = float(f.read().strip())
target_acc = network_accuracies[network_name]
if not almost_equal(target_acc, obtained_acc):
    raise ValueError(
        f"Accuracy mismatch. Obtained: {obtained_acc}, target: {target_acc}"
    )
