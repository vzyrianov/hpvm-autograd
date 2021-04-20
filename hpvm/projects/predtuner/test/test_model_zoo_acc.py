import unittest

import predtuner.model_zoo as net
import torch
from predtuner import TorchApp, accuracy, config_pylogger, get_knobs_from_file
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader

msg_logger = config_pylogger(output_dir="/tmp", verbose=True)


class TestModelZooAcc(unittest.TestCase):
    networks = {
        "lenet_mnist": (net.LeNet, net.MNIST, 2000, 99.65),
        "alexnet_cifar10": (net.AlexNet, net.CIFAR, 500, 78.78),
        "alexnet2_cifar10": (net.AlexNet2, net.CIFAR, 500, 84.76),
        "vgg16_cifar10": (net.VGG16Cifar10, net.CIFAR, 250, 89.22),
        "vgg16_cifar100": (net.VGG16Cifar100, net.CIFAR, 250, 68.42),
        "resnet18_cifar10": (net.ResNet18, net.CIFAR, 250, 89.42),
        "mobilenet_cifar10": (net.MobileNet, net.CIFAR, 250, 84.9),
        "alexnet_imagenet": (net.AlexNetImageNet, net.ImageNet, 20, 55.86),
        # "resnet50_imagenet": (net.ResNet50, net.ImageNet, 10, 71.72),
        "vgg16_imagenet": (net.VGG16ImageNet, net.ImageNet, 5, 68.82),
    }

    def test_all_accuracy(self):
        for name, netinfo in self.networks.items():
            model_cls, dataset_cls, batchsize, target_acc = netinfo
            network: Module = model_cls()
            network.load_state_dict(torch.load(f"model_params/{name}.pth.tar"))
            dataset = dataset_cls.from_file(
                f"model_params/{name}/tune_input.bin",
                f"model_params/{name}/tune_labels.bin",
            )
            tune = DataLoader(dataset, batchsize)
            app = TorchApp("", network, tune, tune, get_knobs_from_file(), accuracy)
            qos, _ = app.empirical_measure_qos_cost({}, False, True)
            self.assertAlmostEqual(qos, target_acc)
