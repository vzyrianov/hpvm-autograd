import abc
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ._logging import PathLike
from .approxapp import ApproxApp, ApproxKnob, ApproxTuner, Config, KnobsT

msg_logger = logging.getLogger(__name__)


class ModeledApp(ApproxApp, abc.ABC):
    """Like `.approxapp.ApproxApp`, but uses a model for QoS/cost measurement.

    To use this class, inherit from it and implement `get_models`,
    `empirical_measure_qos_cost`, and `.approxapp.ApproxApp.name`.
    (This class provides an implementation of `.approxapp.ApproxApp.measure_qos_cost`.)

    :param op_knobs: a mapping from each operator (identified by str) to a list of applicable knobs.
    :type op_knobs: Dict[str, List[ApproxKnob]]
    :param target_device: the target device that this application should be tuned on.
           See `.approxapp.ApproxApp` constructor.
    :type target_device: Optional[str]
    """

    def __init__(
        self, op_knobs: Dict[str, List[ApproxKnob]], target_device: str = None
    ) -> None:
        super().__init__(op_knobs, target_device)
        models = self.get_models()
        self._name_to_model = {m.name: m for m in models}
        if len(self._name_to_model) != len(models):
            raise ValueError("Name conflict in models")
        self._cost_models = {
            model.name: model for model in models if isinstance(model, ICostModel)
        }
        self._qos_models = {
            model.name: model for model in models if isinstance(model, IQoSModel)
        }

    @abc.abstractmethod
    def get_models(self) -> List[Union["ICostModel", "IQoSModel"]]:
        """A list of QoS/Cost prediction models for this application.

        Cost models should inherit from `ICostModel`
        while QoS models should inherit from `IQoSModel`.

        :rtype: List[Union[ICostModel, IQoSModel]]
        """
        pass

    @abc.abstractmethod
    def empirical_measure_qos_cost(
        self, with_approxes: KnobsT, is_test: bool
    ) -> Tuple[float, float]:
        """Empirically measures QoS and cost by actually
        running the program with approximation (as opposed to using model).

        :param with_approxes: The approximation configuration to measure QoS and cost for.
        :param is_test: If True, uses a "test" dataset/mode that is held away from the tuner
               during tuning.
        """

    def measure_qos_cost(
        self,
        with_approxes: KnobsT,
        is_test: bool,
        qos_model: Optional[str] = None,
        cost_model: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Returns the QoS and cost (time, energy, ...) of a given configuration,
        *potentially using models*.

        If either of `cost_model` or `qos_model` is None,
        this will perform empirical measurement once to get the one that is not using a model.
        Otherwise, no empirical measurement will be used.

        Note that when running on test set (``is_test == True``), no modeling is allowed
        (this raises a `ValueError`).

        :param with_approxes: The approximation configuration to measure QoS and cost for.
        :param is_test: If True, uses a "test" dataset/mode that is held away from the tuner
               during tuning; otherwise use "tune" dataset.
        :param qos_model: The QoS model to use in this measurement, keyed by model's name
               (See `IQoSModel.name`).
        :param cost_model: The Cost model to use in this measurement, keyed by model's name
               (See `ICostModel.name`).
        """
        # Testset measurement is always empirical
        if is_test:
            if qos_model is not None or cost_model is not None:
                raise ValueError("Test dataset measurement is always empirical")
            return self.empirical_measure_qos_cost(with_approxes, is_test)
        # Run empirical measurement once if either cost or qos needs it
        qos, cost = None, None
        if qos_model is None or cost_model is None:
            qos, cost = self.empirical_measure_qos_cost(with_approxes, is_test)
        # If we're asked to use some qos_model, overwrite `qos` value
        # even if we already get it from empirical measure (i.e., even if cost_model is None)
        if qos_model is not None:
            if qos_model not in self._qos_models:
                raise ValueError(
                    f'"{qos_model}" is an invalid value for qos_model '
                    f"(choose from {list(self._qos_models.keys())})"
                )
            qos = self._qos_models[qos_model].measure_qos(with_approxes)
        # Same goes for cost
        if cost_model is not None:
            if cost_model not in self._cost_models:
                raise ValueError(
                    f'"{cost_model}" is an invalid value for cost_model '
                    f"(choose from {list(self._cost_models.keys())})"
                )
            cost = self._cost_models[cost_model].measure_cost(with_approxes)
        assert type(qos) is float and type(cost) is float
        return qos, cost

    def get_tuner(self) -> "ApproxModeledTuner":
        """Sets up an ApproxTuner instance which the user can directly call
        `tune()` on with opentuner parameters.

        This returns an `ApproxModeledTuner`, different from `.approxapp.ApproxApp.get_tuner`
        which returns an `ApproxTuner`.

        :rtype: ApproxModeledTuner
        """

        return ApproxModeledTuner(self)

    def init_model(self, model_name: str):
        self._name_to_model[model_name]._init()


class ICostModel(abc.ABC):
    """Abstract base class for models that provide cost prediction."""

    def __init__(self) -> None:
        self._inited = False

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of model."""
        pass

    @abc.abstractmethod
    def measure_cost(self, with_approxes: KnobsT) -> float:
        """Predict the cost of application.

        :param with_approxes: The configuration to predict cost for.
        """
        pass

    def _init(self):
        """Initialize the model before the first prediction task (profiling, etc.)"""
        self._inited = True


class IQoSModel(abc.ABC):
    """Abstract base class for models that provide QoS prediction."""

    def __init__(self) -> None:
        self._inited = False

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of model."""
        pass

    @abc.abstractmethod
    def measure_qos(self, with_approxes: KnobsT) -> float:
        """Predict the QoS of application.

        :param with_approxes: The configuration to predict QoS for.
        """
        pass

    def _init(self):
        """Initialize the model before the first prediction task (profiling, etc.)"""
        self._inited = True


class LinearCostModel(ICostModel):
    """Weighted linear cost predictor based on cost of each operator.

    This predictor compute a weighted sum over
    the cost of each operator and the speedup of each knob on that operator.

    :param app: The `ModeledApp` to predict cost for.
    :param op_costs: A mapping from operator name to its (baseline) cost.
    :param knob_speedups: A mapping from knob name to its (expected) speedup.
    """

    def __init__(
        self,
        app: ModeledApp,
        op_costs: Dict[str, float],
        knob_speedups: Dict[str, float],
    ) -> None:
        import numpy as np
        import pandas as pd

        super().__init__()
        self.app = app
        knob_cost_factor_v = 1 / np.array(list(knob_speedups.values()))
        layer_cost_v = np.array(list(op_costs.values()))
        costs = np.outer(layer_cost_v, knob_cost_factor_v)
        self.cost_df = pd.DataFrame(
            costs, index=op_costs.keys(), columns=knob_speedups.keys(), dtype=float
        )

    @property
    def name(self) -> str:
        return "cost_linear"

    def measure_cost(self, with_approxes: KnobsT) -> float:
        with_approxes = self.app.add_baseline_to_knobs(with_approxes)
        return float(
            sum(self.cost_df.loc[layer, knob] for layer, knob in with_approxes.items())
        )


class QoSModelP1(IQoSModel):
    """QoS model `P1` in ApproxTuner.

    :param app: The `ModeledApp` to predict QoS for.
    :param tensor_output_getter: A function that can run the
           tensor-based application with a config and return a single tensor result.

           Note that here we require the return value to be a PyTorch tensor.

    :param qos_metric: A function that compute a QoS level from the return value
           of `tensor_output_getter`.
    :param storage: A file of PyTorch format to store this model into, if the file doesn't exist,
           or load the model from if the file exists.
           If not given, the model will not be stored.
    """

    def __init__(
        self,
        app: ModeledApp,
        tensor_output_getter: Callable[[KnobsT], torch.Tensor],
        qos_metric: Callable[[torch.Tensor], float],
        storage: PathLike = None,
    ) -> None:
        from torch.nn.functional import softmax

        super().__init__()
        self.app = app
        self.output_f = tensor_output_getter
        self.qos_metric = qos_metric
        self.storage = Path(storage) if storage else None
        self.delta_tensors = {
            op: {k.name: None for k in self.app.knobs} for op in self.app.ops
        }
        self.baseline_tensor = None

    @property
    def name(self) -> str:
        return "qos_p1"

    def measure_qos(self, with_approxes: KnobsT) -> float:
        assert self.baseline_tensor is not None
        with_approxes = self.app.add_baseline_to_knobs(with_approxes)
        delta_sum = sum(
            [self.delta_tensors[op][knob] for op, knob in with_approxes.items()]
        )
        ret = delta_sum + self.baseline_tensor
        return float(self.qos_metric(ret))

    def _init(self):
        if self.storage and self.storage.is_file():
            self.delta_tensors, self.baseline_tensor = torch.load(self.storage)
        dt = self.delta_tensors
        btensor = self.baseline_tensor = self.output_f({})
        updated = False
        for op, knob in barred_ravel_knobs(self.app):
            if dt[op][knob] is not None:
                continue
            updated = True
            delta_tensor = self.output_f({op: knob}) - btensor
            dt[op][knob] = delta_tensor
        if self.storage and updated:
            os.makedirs(self.storage.parent, exist_ok=True)
            torch.save((dt, btensor), self.storage)
        super()._init()


class QoSModelP2(IQoSModel):
    """QoS model `P1` in ApproxTuner.

    :param app: The `ModeledApp` to predict QoS for.
    :param storage: A JSON file to store this model into, if the file doesn't exist,
           or load the model from if the file exists.
           If not given, the model will not be stored.
    """

    def __init__(self, app: ModeledApp, storage: PathLike = None) -> None:
        super().__init__()
        self.app = app
        self.storage = Path(storage) if storage else None
        self.qos_df = None
        self.baseline_qos = None

    @property
    def name(self) -> str:
        return "qos_p2"

    def _empirical_measure_qos(self, with_approxes: KnobsT) -> float:
        """An internal QoS-measuring method.

        The point is P2 queries some QoS results and caches them before tuning starts,
        and then defines a `measure_qos` that doesn't run the application during tuning
        (to reduce overhead).
        """
        qos, _ = self.app.empirical_measure_qos_cost(with_approxes, False)
        return qos

    def measure_qos(self, with_approxes: KnobsT) -> float:
        assert self.baseline_qos is not None and self.qos_df is not None
        with_approxes = self.app.add_baseline_to_knobs(with_approxes)
        delta_qoses = (
            np.array([self.qos_df.loc[kv] for kv in with_approxes.items()])
            - self.baseline_qos
        )
        ret = delta_qoses.sum() + self.baseline_qos
        assert not np.isnan(ret)
        return float(ret)

    def _init(self):
        if self.storage and self.storage.is_file():
            self._load(self.storage)
        else:
            knob_names = [k.name for k in self.app.knobs]
            self.qos_df = pd.DataFrame(index=self.app.ops, columns=knob_names)
            self.qos_df = self.qos_df.where(pd.notnull(self.qos_df), None)
            self.baseline_qos = self._empirical_measure_qos({})
        df = self.qos_df
        for op, knob in barred_ravel_knobs(self.app):
            if df.loc[op, knob] is not None:
                continue
            df.loc[op, knob] = self._empirical_measure_qos({op: knob})
        if self.storage and not self.storage.is_file():
            self._save(self.storage)
        super()._init()

    def _load(self, path: Path):
        with path.open() as f:
            data = json.load(f)
        if "app_name" in data:
            name = data["app_name"]
            if self.app.name != name:
                msg_logger.error(
                    f'Profile at {path} belongs to app "{name}" '
                    f"while our app is {self.app.name}"
                )
        else:
            msg_logger.warning("Loaded profile does not have app name identifier")
        msg_logger.info(f"Model {self.name} loaded saved model at {path}")
        self.qos_df = pd.DataFrame(data["df"])
        self.baseline_qos = float(data["bqos"])

    def _save(self, path: Path):
        if not path.parent.is_dir():
            os.makedirs(path.parent)
        with path.open("w") as f:
            json.dump(
                {
                    "app_name": self.app.name,
                    "df": self.qos_df.to_dict(),
                    "bqos": self.baseline_qos,
                },
                f,
            )


class ValConfig(Config):
    """An `.approxapp.Config` that also optionally stores the "validation QoS".

    Validation QoS is the empirically measured QoS in the "validation phase"
    at the end of tuning (see `ApproxModeledTuner.tune`).

    :param qos: The maybe-predicted QoS of this config.
           (If tuning is empirical then this is empirical, not predicted, QoS.)
           This is in contrast to `Config.qos`, which is always empirically measured on tuning dataset.
    :param cost: The *relative* cost (time, energy, etc.) of this config
           compared to the baseline config. This is essentially :math:`1 / speedup`.
    :param knobs: The op-knob mapping in this configuration.
    :param test_qos: The empirically measured QoS of this config on test mode.
    :param validated_qos: The empirically measured QoS of this config on tuning mode,
           in the validation phase. See `ApproxModeledTuner.tune`.
    """

    def __init__(
        self,
        qos: float,
        cost: float,
        knobs: KnobsT,
        test_qos: Optional[float] = None,
        validated_qos: Optional[float] = None,
    ) -> None:
        super().__init__(qos, cost, knobs, test_qos)
        self.validated_qos = validated_qos


class ApproxModeledTuner(ApproxTuner):
    app: ModeledApp

    def tune(
        self,
        max_iter: int,
        qos_tuner_threshold: float,
        qos_keep_threshold: Optional[float] = None,
        is_threshold_relative: bool = False,
        take_best_n: Optional[int] = None,
        test_configs: bool = True,
        validate_configs: Optional[bool] = None,
        cost_model: Optional[str] = None,
        qos_model: Optional[str] = None,
    ) -> List[ValConfig]:
        """Runs a tuning session.

        :param max_iter: Number of iterations to use in tuning.
        :param qos_tuner_threshold: The QoS threshold that the tuner should aim for.
               QoS is assumed to be a higher-better quantity.
               This should be slightly tighter than `qos_keep_threshold`
               to account for extra error when running on test dataset.
        :param qos_keep_threshold: The QoS threshold beyond which we will keep the configuration.
               By default it is equal to `qos_keep_threshold`.
        :param is_threshold_relative: If True, the actual thresholds are considered to be
               ``baseline_qos - given_threshold``.
               This applies to `qos_tuner_threshold` and `qos_keep_threshold`.
        :param take_best_n: Take the best :math:`n` configurations after tuning.
               "Best" is defined as the configurations closest to the pareto curve
               of the QoS-cost tradeoff space.
               If `take_best_n` is None, only the configurations strictly on the
               pareto curve are taken.
        :param test_configs: If True, runs the configs on the test dataset,
               filter the taken configs by `qos_keep_threshold`,
               and fill the `test_qos` field of `ValConfig`.
        :param validate_configs: If True, runs a validation step that empirically measures
               the QoS of configs, filter the taken configs by `qos_keep_threshold`,
               and fill the `validated_qos` field of `ValConfig`.
        :param cost_model: The cost model to use for this tuning session.
        :param qos_model: The QoS model to use for this tuning session.
               This and `cost_model` are relayed down the line to `ModeledApp.measure_qos_cost`.
        """

        qos_desc = (
            "no model for qos" if qos_model is None else f'qos model "{qos_model}"'
        )
        cost_desc = (
            "no model for cost" if cost_model is None else f'cost model "{cost_model}"'
        )
        msg_logger.info("Starting tuning with %s and %s", qos_desc, cost_desc)
        if qos_model is not None:
            msg_logger.info("Initializing qos model %s", qos_model)
            self.app.init_model(qos_model)
        if cost_model is not None:
            msg_logger.info("Initializing cost model %s", cost_model)
            self.app.init_model(cost_model)
        super().tune(
            max_iter=max_iter,
            qos_tuner_threshold=qos_tuner_threshold,
            qos_keep_threshold=qos_keep_threshold,
            is_threshold_relative=is_threshold_relative,
            take_best_n=take_best_n,
            test_configs=False,  # Test configs below by ourselves
            app_kwargs={"cost_model": cost_model, "qos_model": qos_model},
        )
        if validate_configs is None and qos_model is not None:
            msg_logger.info(
                'Validating configurations due to using qos model "%s"', qos_model
            )
            self._update_configs_(self.best_configs_prefilter, False)
        elif validate_configs:
            msg_logger.info("Validating configurations as user requested")
            self._update_configs_(self.best_configs_prefilter, False)
        if test_configs:
            msg_logger.info("Calibrating configurations on test inputs")
            self._update_configs_(self.best_configs_prefilter, True)
        self.best_configs = self._filter_configs(self.best_configs_prefilter)
        return self.best_configs

    def _update_configs_(self, configs: List[ValConfig], test_mode: bool):
        from tqdm import tqdm

        if not configs:
            msg_logger.info("No configurations found.")
            return
        ret_configs = []
        total_error = 0
        for cfg in tqdm(configs, leave=False):
            qos, _ = self.app.measure_qos_cost(cfg.knobs, test_mode)
            if test_mode:
                assert cfg.test_qos is None
                cfg.test_qos = qos
                msg_logger.debug(f"Test: {cfg.qos} (mean) -> {qos} (mean)")
            else:
                assert cfg.validated_qos is None
                cfg.validated_qos = qos
                msg_logger.debug(f"Validation: {cfg.qos} (mean) -> {qos} (mean)")
            total_error += abs(cfg.qos - qos)
        mean_err = total_error / len(configs)
        dataset_name = "test" if test_mode else "tune"
        msg_logger.info(
            "QoS changed by %f on %s dataset (mean abs diff)", mean_err, dataset_name
        )

    def _filter_configs(self, configs: List[ValConfig]):
        ret_configs = [
            cfg
            for cfg in configs
            if (not cfg.validated_qos or cfg.validated_qos >= self.tune_keep_threshold)
            and cfg.test_qos >= self.test_keep_threshold
        ]
        msg_logger.info(
            "%d of %d configs remain after validation and testing",
            len(ret_configs),
            len(configs),
        )
        return ret_configs

    def plot_configs(
        self,
        show_qos_loss: bool = False,
        connect_best_points: bool = False,
    ) -> plt.Figure:
        """Plots 1 to 3 QoS-vs-speedup scatter plot of configurations.

        All kept configurations and all "best" configurations (before test-set filtering if any)
        are always plotted in the first subplot.

        If there was a validation phase during tuning,
        the second subplot contains the "best" configurations plotted twice,
        with predicted and empirically measured QoS (on tune set) respectively.

        If both validation and test-set filtering was used,
        the last subplot contains the "best" configurations
        with *empirically measured* tune-set and test-set QoS loss respectively.

        :param show_qos_loss: If True, uses the loss of QoS (compared to the baseline)
               instead of the absolute QoS in the first 2 graphs.
               *This does not apply to the third graph* if it exists,
               which always use QoS loss for ease of comparison.
        """

        if not self.tuned:
            raise RuntimeError(
                f"No tuning session has been run; call self.tune() first."
            )
        dot_format = "-o" if connect_best_points else "o"
        # Without `ax` argument, this function returns if we can
        # do the second/third plot or not.
        # plot_test_phase returns True implies plot_validation_phase returning True.
        val_phase = self.plot_validation_phase()
        test_phase = self.plot_test_phase()
        n_subplots = 1 + int(val_phase) + int(test_phase)
        fig, axes = plt.subplots(
            1, n_subplots, squeeze=False, figsize=(6 + 4 * n_subplots, 6), dpi=300
        )

        i = 1
        self.plot_kept_and_best(axes[0, 0], show_qos_loss)
        if val_phase:
            ax = axes[0, i]
            self.plot_validation_phase(ax, show_qos_loss, dot_format)
            i += 1
        if test_phase:
            ax = axes[0, i]
            tuneset_key = "validated_qos" if val_phase else "qos"
            self.plot_test_phase(ax, dot_format, tuneset_key)
            i += 1
        fig.tight_layout()
        return fig

    def plot_validation_phase(
        self, ax: plt.Axes = None, show_qos_loss: bool = False, dot_format: str = "o"
    ):
        configs = self.best_configs_prefilter
        validated = [conf.validated_qos is not None for conf in configs]
        can_plot = all(validated)
        if not ax:
            return can_plot
        assert can_plot
        pred_x, pred_y = self._config_qos_speedups(configs, "qos", show_qos_loss, False)
        measured_x, measured_y = self._config_qos_speedups(
            configs, "validated_qos", show_qos_loss, False
        )
        ax.plot(pred_x, pred_y, dot_format, label="Predicted QoS")
        ax.plot(measured_x, measured_y, dot_format, label="Validated QoS")
        self._set_xy_limit(ax, show_qos_loss)
        if show_qos_loss:
            ax.set_xlabel("QoS Loss (tune dataset)")
            rthres = self.baseline_tune_qos - self.tune_keep_threshold
            self._draw_qos_line(ax, rthres, f"Relative threshold: {rthres:.2f}")
        else:
            ax.set_xlabel("QoS (tune dataset)")
        ax.set_ylabel("Speedup (x)")
        ax.legend()

    @classmethod
    def _get_config_class(cls) -> Type[Config]:
        return ValConfig


def barred_ravel_knobs(app: ApproxApp) -> Iterator[Tuple[str, str]]:
    """Flattens op_knobs of app to a list of (layer, knob) pairs while showing 2 levels of
    progress bar."""

    from tqdm import tqdm

    bar1 = tqdm(app.op_knobs.items(), leave=None)
    for op_name, knobs in bar1:
        bar1.set_postfix(op=op_name)
        bar2 = tqdm(knobs, leave=None)
        for knob in bar2:
            bar2.set_postfix(knob=knob.name)
            yield op_name, knob.name
