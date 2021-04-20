import site
from pathlib import Path

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

site.addsitedir(Path(__file__).parent.parent.absolute().as_posix())
from predtuner import TorchApp, accuracy, config_pylogger, get_knobs_from_file
from predtuner.model_zoo import CIFAR, VGG16Cifar10

# Set up logger to put log file in /tmp
msg_logger = config_pylogger(output_dir="/tmp", verbose=True)

# Load "tuning" dataset and "test" dataset,
# and use only the first 500 images from each dataset as an example
# TODO: you should use all (5000) images for actual tuning.
prefix = Path("model_params/vgg16_cifar10")
tune_set = CIFAR.from_file(prefix / "tune_input.bin", prefix / "tune_labels.bin")
tune_loader = DataLoader(tune_set, batch_size=500)
test_set = CIFAR.from_file(prefix / "test_input.bin", prefix / "test_labels.bin")
test_loader = DataLoader(test_set, batch_size=500)

# Load checkpoint for VGG16 (CIFAR10)
module = VGG16Cifar10()
module.load_state_dict(torch.load("model_params/vgg16_cifar10.pth.tar"))
app = TorchApp(
    "TestTorchApp",  # name -- can be anything
    module,
    tune_loader,
    test_loader,
    get_knobs_from_file(),  # default knobs -- see "predtuner/approxes/default_approx_params.json"
    accuracy,  # the QoS metric to use -- classification accuracy
    # Where to serialize prediction models if they are used
    # For example, if you use p1 (see below), this will leave you a
    # tuner_results/vgg16_cifar10/p1.pkl
    # which can be quickly reloaded the next time you do tuning with
    model_storage_folder="tuner_results/vgg16_cifar10",
)
# This is how to measure baseline accuracy -- {} means no approximation
baseline, _ = app.measure_qos_cost({}, False)
# Get a tuner object and start tuning!
tuner = app.get_tuner()
tuner.tune(
    max_iter=1000,  # TODO: In practice, use at least 5000, or 10000
    qos_tuner_threshold=2.0,  # QoS threshold to guide tuner into
    qos_keep_threshold=3.0,  # QoS threshold for which we actually keep the configurations
    is_threshold_relative=True,  # Thresholds are relative to baseline -- baseline_acc - 2.1
    take_best_n=20,  # Take the best 20 configs (not just the "strictly" best ones)
    cost_model="cost_linear",  # Use linear performance predictor
    qos_model="qos_p1",  # Use P1 QoS predictor
)
# Save configs here when you're done
tuner.dump_configs("tuner_results/vgg16_cifar10_configs.json")
fig = tuner.plot_configs(show_qos_loss=True)
fig.savefig("tuner_results/vgg16_cifar10_configs.png")