import unittest

import torch
from predtuner.model_zoo import CIFAR, VGG16Cifar10
from predtuner import TorchApp, accuracy, config_pylogger, get_knobs_from_file
from torch.nn import Conv2d, Linear
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset

msg_logger = config_pylogger(output_dir="/tmp", verbose=True)


class TorchAppSetUp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = CIFAR.from_file(
            "model_params/vgg16_cifar10/tune_input.bin",
            "model_params/vgg16_cifar10/tune_labels.bin",
        )
        cls.dataset = Subset(dataset, range(100))
        cls.module = VGG16Cifar10()
        cls.module.load_state_dict(torch.load("model_params/vgg16_cifar10.pth.tar"))
        cls.app_args = {
            "app_name": "TestTorchApp",
            "module": cls.module,
            "tune_dataloader": DataLoader(cls.dataset, batch_size=500),
            "test_dataloader": DataLoader(cls.dataset, batch_size=500),
            "knobs": get_knobs_from_file(),
            "tensor_to_qos": accuracy,
        }
        cls.app = TorchApp(**cls.app_args)


class TestTorchAppTuning(TorchAppSetUp):
    def test_knobs(self):
        n_knobs = {op: len(ks) for op, ks in self.app.op_knobs.items()}
        self.assertEqual(len(n_knobs), 34)
        for op_name, op in self.app.midx.name_to_module.items():
            nknob = 56 if isinstance(op, Conv2d) else 2
            self.assertEqual(n_knobs[op_name], nknob)
        self.assertEqual(self.app.baseline_knob.name, "11")

    def test_cpu_knobs(self):
        app = TorchApp(**self.app_args, target_device="cpu")
        n_knobs = {op: len(ks) for op, ks in app.op_knobs.items()}
        for op_name, op in app.midx.name_to_module.items():
            nknob = 28 if isinstance(op, Conv2d) else 1
            self.assertEqual(n_knobs[op_name], nknob)
        self.assertEqual(app.baseline_knob.name, "11")

    def test_gpu_knobs(self):
        app = TorchApp(**self.app_args, target_device="gpu")
        n_knobs = {op: len(ks) for op, ks in app.op_knobs.items()}
        for op_name, op in app.midx.name_to_module.items():
            nknob = 28 if isinstance(op, Conv2d) else 1
            self.assertEqual(n_knobs[op_name], nknob)
        self.assertEqual(app.baseline_knob.name, "12")

    def test_baseline_qos(self):
        qos, _ = self.app.measure_qos_cost({}, False)
        self.assertAlmostEqual(qos, 93.0)

    def test_tuning_relative_thres(self):
        baseline, _ = self.app.measure_qos_cost({}, False)
        tuner = self.app.get_tuner()
        tuner.tune(100, 3.0, 3.0, True, 10)
        for conf in tuner.kept_configs:
            self.assertTrue(conf.qos > baseline - 3.0)
        if len(tuner.kept_configs) >= 10:
            self.assertEqual(len(tuner.best_configs), 10)

    def test_enum_models(self):
        self.assertSetEqual(
            set(model.name for model in self.app.get_models()),
            {"cost_linear", "qos_p1", "qos_p2"},
        )


class TestTorchAppTunerResult(TorchAppSetUp):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.baseline, _ = cls.app.measure_qos_cost({}, False)
        cls.tuner = cls.app.get_tuner()
        cls.tuner.tune(100, cls.baseline - 3.0)

    def test_results_qos(self):
        configs = self.tuner.kept_configs
        for conf in configs:
            self.assertTrue(conf.qos > self.baseline - 3.0)

    def test_pareto(self):
        configs = self.tuner.best_configs
        for c1 in configs:
            self.assertFalse(
                any(c2.qos > c1.qos and c2.cost > c1.cost for c2 in configs)
            )

    def test_dummy_testset(self):
        configs = self.tuner.best_configs
        for c in configs:
            self.assertAlmostEqual(c.test_qos, c.qos)


class TestModeledTuning(TorchAppSetUp):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.baseline, _ = cls.app.measure_qos_cost({}, False)

    def test_qos_p1(self):
        tuner = self.app.get_tuner()
        tuner.tune(
            100,
            3.0,
            is_threshold_relative=True,
            cost_model="cost_linear",
            qos_model="qos_p1",
        )

    def test_qos_p2(self):
        tuner = self.app.get_tuner()
        tuner.tune(
            100,
            3.0,
            is_threshold_relative=True,
            cost_model="cost_linear",
            qos_model="qos_p2",
        )


class TestModelSaving(TorchAppSetUp):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.baseline, _ = cls.app.measure_qos_cost({}, False)
        cls.model_path = "/tmp/test_models"
        app = cls.get_app()
        app.init_model("qos_p1")
        app.init_model("qos_p2")
        cls.app = cls.get_app()

    @classmethod
    def get_app(cls):
        return TorchApp(
            "TestTorchApp",
            cls.module,
            DataLoader(cls.dataset, batch_size=500),
            DataLoader(cls.dataset, batch_size=500),
            get_knobs_from_file(),
            accuracy,
            model_storage_folder=cls.model_path,
        )

    def test_loading_p1(self):
        self.app.get_tuner().tune(
            100,
            3.0,
            is_threshold_relative=True,
            cost_model="cost_linear",
            qos_model="qos_p1",
        )

    def test_loading_p2(self):
        self.app.get_tuner().tune(
            100,
            3.0,
            is_threshold_relative=True,
            cost_model="cost_linear",
            qos_model="qos_p2",
        )
