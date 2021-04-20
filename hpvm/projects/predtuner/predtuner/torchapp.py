import abc
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data.dataloader import DataLoader

from ._logging import PathLike
from .approxapp import ApproxKnob, KnobsT
from .modeledapp import (
    ICostModel,
    IQoSModel,
    LinearCostModel,
    ModeledApp,
    QoSModelP1,
    QoSModelP2,
)
from .torchutil import ModuleIndexer, get_summary, move_to_device_recursively


class TorchApproxKnob(ApproxKnob):
    """Defines an approximation knob that knows
    its own expected speedup ratio and what Modules it can apply to,
    and can be applied to a torch.nn.Module to return an approximated Module."""

    @property
    @abc.abstractmethod
    def deterministic(self) -> bool:
        """Returns true if approx knob does not contain randomness."""
        pass

    @property
    @abc.abstractmethod
    def expected_speedup(self) -> float:
        """The speedup this knob is expected to provide. Used for cost prediction."""
        pass

    @abc.abstractmethod
    def is_applicable(self, op: Module) -> bool:
        """Returns True if this knob can be applied to this Module.
        
        :param op: the module to check availability for.
        :type op: torch.nn.Module
        :rtype: torch.nn.Module
        """
        pass

    @abc.abstractmethod
    def apply(self, op: Module) -> Module:
        """Applies knob to a Module and returns an approximated Module.
        
        :param op: the module to apply approximation on.
        :type op: torch.nn.Module
        :rtype: torch.nn.Module
        """
        pass


_default_device = f"cuda" if torch.cuda.is_available() else "cpu"


class TorchApp(ModeledApp, abc.ABC):
    r"""Adaptor for approximable PyTorch Modules with tensor output.

    A TorchApp stores the PyTorch Module, datasets for tuning and calibration,
    set of available `TorchApproxKnob` each of which may be applied to some layer in the Module,
    and the quality of service (QoS) metric of application (e.g., accuracy).
    It provides empirical tuning and predictive tuning capability,
    automatically supporting `.modeledapp.LinearCostModel`,
    `.modeledapp.QoSModelP1`, and `.modeledapp.QoSModelP2`.

    In contrast to `.approxapp.ApproxApp` and `.modeledapp.ModeledApp`,
    there should be no need to inherit from `TorchApp` in most use cases.

    :param app_name: Name of the application, which is used as an identifier in tuning sessions, etc.
    :param module: The PyTorch module to tune.
    :param tune_dataloader: A `torch.utils.data.Dataset` dataset to use as inputs to module during tuning.
    :param test_dataloader: A `torch.utils.data.Dataset` dataset used for QoS testing
           (see `test_configs` parameter of `ApproxModeledTuner.tune`).
    :param knobs: A set of `TorchApproxKnob` to be considered. Each knob has an `is_applicable()` method
           which is used to determine which layer it can apply to.
           `.approxes.get_knobs_from_file` returns a set of builtin knobs that will exactly fit here.
    :param tensor_to_qos: QoS metric function which computes QoS from the module's output.
           `.torchutil.accuracy` computes the classification accuracy which can be applied here.
    :param combine_qos: A function to combine each batch's QoS into one value.
           When QoS is Classification Accuracy, this will most likely be `numpy.mean`
           (which is the default value).
    :param target_device: The target device that this application should be tuned on.
    :param torch_device: The PyTorch device where the model inference is run on.
           This device should be able to run the implementations of the knobs
           available for this app on `target_device`.
    :param model_storage_folder: A folder to store the serialized QoS models into.
           `QoSModelP1` will be serialized into ``model_storage_folder / "p1.pkl"``,
           and `QoSModelP2` into ``model_storage_folder / "p2.json"``.
    """

    def __init__(
        self,
        app_name: str,
        module: Module,
        tune_dataloader: DataLoader,
        test_dataloader: DataLoader,
        knobs: Set[TorchApproxKnob],
        tensor_to_qos: Callable[[torch.Tensor, Any], float],
        combine_qos: Callable[[np.ndarray], float] = np.mean,
        target_device: str = None,
        torch_device: Union[torch.device, str] = _default_device,
        model_storage_folder: Optional[PathLike] = None,
    ) -> None:
        self.app_name = app_name
        self.module = module
        self.tune_loader = tune_dataloader
        self.test_loader = test_dataloader
        self.name_to_knob = {
            k.name: k for k in self._check_and_filter_knob(knobs, target_device)
        }
        self.tensor_to_qos = tensor_to_qos
        self.combine_qos = combine_qos
        self.device = torch_device
        self.model_storage = (
            Path(model_storage_folder) if model_storage_folder else None
        )

        self.module = self.module.to(torch_device)
        self.midx = ModuleIndexer(module)
        self._op_costs = {}
        op_knobs = {}
        self._knob_speedups = {k.name: k.expected_speedup for k in knobs}
        modules = self.midx.name_to_module
        summary = get_summary(self.module, (self._sample_input(),))
        for op_name, op in modules.items():
            this_knobs = [
                knob for knob in self.name_to_knob.values() if knob.is_applicable(op)
            ]
            assert this_knobs
            op_knobs[op_name] = this_knobs
            self._op_costs[op_name] = summary.loc[op_name, "flops"]

        # Init parent class last
        super().__init__(op_knobs, target_device)

    @property
    def name(self) -> str:
        """Returns the name of application."""
        return self.app_name

    def get_models(self) -> List[Union[ICostModel, IQoSModel]]:
        """Returns a list of predictive tuning models.

        TorchApp in particular derives 1 performance model (LinearCostModel)
        and 2 QoS models (QoSModelP1, QoSModelP2) automatically.
        """

        def batched_valset_qos(tensor_output: torch.Tensor):
            dataset_len = len(self.tune_loader.dataset)
            assert len(tensor_output) == dataset_len
            begin = 0
            qoses = []
            for _, target in self.tune_loader:
                end = begin + len(target)
                target = move_to_device_recursively(target, self.device)
                qos = self.tensor_to_qos(tensor_output[begin:end], target)
                qoses.append(qos)
                begin = end
            return self.combine_qos(np.array(qoses))

        p1_storage = self.model_storage / "p1.pkl" if self.model_storage else None
        p2_storage = self.model_storage / "p2.json" if self.model_storage else None
        return [
            LinearCostModel(self, self._op_costs, self._knob_speedups),
            QoSModelP1(
                self, self._get_raw_output_valset, batched_valset_qos, p1_storage
            ),
            QoSModelP2(self, p2_storage),
        ]

    @torch.no_grad()
    def empirical_measure_qos_cost(
        self, with_approxes: KnobsT, is_test: bool, progress: bool = False
    ) -> Tuple[float, float]:
        """Measure the QoS and performance of Module with given approximation
        empirically (i.e., by running the Module on the dataset)."""

        from time import time
        from tqdm import tqdm

        dataloader = self.test_loader if is_test else self.tune_loader
        if progress:
            dataloader = tqdm(dataloader)
        approxed = self._apply_knobs(with_approxes)
        qoses = []

        time_begin = time()
        for inputs, targets in dataloader:
            inputs = move_to_device_recursively(inputs, self.device)
            targets = move_to_device_recursively(targets, self.device)
            outputs = approxed(inputs)
            qoses.append(self.tensor_to_qos(outputs, targets))
        time_end = time()
        qos = float(self.combine_qos(np.array(qoses)))
        return qos, time_end - time_begin

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        module_class_name = type(self.module).__name__
        return (
            f'{class_name}"{self.name}"(module={module_class_name}, '
            f"num_op={len(self.op_knobs)}, num_knob={len(self.name_to_knob)})"
        )

    @torch.no_grad()
    def _get_raw_output_valset(self, with_approxes: KnobsT):
        approxed = self._apply_knobs(with_approxes)
        all_outputs = []
        for inputs, _ in self.tune_loader:
            inputs = move_to_device_recursively(inputs, self.device)
            outputs = approxed(inputs)
            all_outputs.append(outputs)
        return torch.cat(all_outputs, dim=0)

    @staticmethod
    def _check_and_filter_knob(
        knobs: Set[TorchApproxKnob], device: Optional[str]
    ) -> Set[TorchApproxKnob]:
        baseline = ApproxKnob.unique_baseline(knobs)
        if baseline not in knobs:
            knobs.add(baseline)
        if not device:
            return knobs
        return {knob for knob in knobs if knob.exists_on_device(device)}

    def _apply_knobs(self, knobs: KnobsT) -> Module:
        import copy

        module_indexer = copy.deepcopy(self.midx)
        for op_name, knob_name in knobs.items():
            knob = self.name_to_knob[knob_name]
            module_indexer[op_name] = knob.apply(module_indexer[op_name])
        return module_indexer.module

    def _sample_input(self):
        inputs, _ = next(iter(DataLoader(self.tune_loader.dataset, batch_size=1)))
        return inputs.to(self.device)
