import site
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

site.addsitedir(Path(__file__).absolute().parent.parent)
from predtuner.model_zoo import CIFAR, VGG16Cifar10
from predtuner import TorchApp, accuracy, config_pylogger, get_knobs_from_file

msg_logger = config_pylogger(output_dir="tuner_results/logs", verbose=True)

tune_set = CIFAR.from_file(
    "model_params/vgg16_cifar10/tune_input.bin",
    "model_params/vgg16_cifar10/tune_labels.bin",
)
tune_loader = DataLoader(Subset(tune_set, range(500)), batch_size=500)
test_set = CIFAR.from_file(
    "model_params/vgg16_cifar10/test_input.bin",
    "model_params/vgg16_cifar10/test_labels.bin",
)
test_loader = DataLoader(Subset(test_set, range(500)), batch_size=500)
module = VGG16Cifar10()
module.load_state_dict(torch.load("model_params/vgg16_cifar10.pth.tar"))
app = TorchApp(
    "TestTorchApp",
    module,
    tune_loader,
    test_loader,
    get_knobs_from_file(),
    accuracy,
    model_storage_folder="tuner_results/vgg16_cifar10",
)
baseline, _ = app.measure_qos_cost({}, False)
tuner = app.get_tuner()
tuner.tune(500, 2.1, 3.0, True, 20, cost_model="cost_linear", qos_model="qos_p1")
tuner.dump_configs("tuner_results/test/configs.json")
fig = tuner.plot_configs(show_qos_loss=True)
fig.savefig("tuner_results/test/configs.png", dpi=300)
