import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .approxapp import ApproxKnob, KnobsT
from .modeledapp import (
    ICostModel,
    IQoSModel,
    LinearCostModel,
    ModeledApp,
    QoSModelP1,
    QoSModelP2,
    ValConfig,
)
from .torchutil import accuracy

PathLike = Union[str, Path]


class PipedBinaryApp(ModeledApp):
    def __init__(
        self,
        app_name: str,
        binary_path: PathLike,
        metadata_path: PathLike,
        base_dir: PathLike = None,
        qos_relpath: PathLike = "final_accuracy",
        target_device: str = None,
        model_storage_folder: Optional[PathLike] = None,
    ):
        self.app_name = app_name
        self.binary_path = Path(binary_path).absolute()
        self.base_dir = (
            self.binary_path.parent if base_dir is None else Path(base_dir).absolute()
        )
        self.qos_file = self.base_dir / qos_relpath
        (
            self.op_costs,
            op_knobs,
            self.knob_speedup,
            self.tune_labels,
            self.conf_file,
            self.fifo_r_file,
            self.fifo_w_file,
        ) = self._read_metadata(metadata_path)
        self._op_order = {v: i for i, v in enumerate(op_knobs.keys())}
        self.model_storage = (
            Path(model_storage_folder) if model_storage_folder else None
        )
        if not self.binary_path.is_file():
            raise RuntimeError(f"Binary file {self.binary_path} not found")

        super().__init__(op_knobs, target_device)  # Init here
        self.knob_exporter = HPVMConfigBuilder(list(op_knobs.keys()))
        self.process = None
        self._invoke_binary()

        # Measure our own test set accuracy for config export purpose
        self.base_test_qos, _ = self.empirical_measure_qos_cost({}, True)

    def dump_hpvm_configs(
        self,
        configs: Sequence[ValConfig],
        output: PathLike = None,
        add_baseline_conf: bool = True,
    ):
        configs = list(configs)
        tqos = self.base_test_qos
        if add_baseline_conf:
            base_knobs = self.add_baseline_to_knobs({})
            base_conf = ValConfig(tqos, 1.0, base_knobs, tqos, tqos)
            configs = [base_conf] + configs
        configs = sorted(configs, key=lambda c: c.speedup)
        conf_str = self.knob_exporter.configs_to_str(configs, tqos)
        if output is None:
            return conf_str
        output_p = Path(output)
        os.makedirs(output_p.parent, exist_ok=True)
        with output_p.open("w") as f:
            f.write(conf_str)

    @property
    def name(self) -> str:
        """Name of application. Acts as an identifier in many places, so
        the user should try to make it unique."""
        return self.app_name

    def empirical_measure_qos_cost(
        self, with_approxes: KnobsT, is_test: bool
    ) -> Tuple[float, float]:
        from time import time

        time_begin = time()
        _, qos = self._run_on_knobs(with_approxes, is_test)
        time_end = time()
        return qos, time_end - time_begin

    def get_models(self) -> List[Union["ICostModel", "IQoSModel"]]:
        p1_storage = self.model_storage / "p1.pkl" if self.model_storage else None
        p2_storage = self.model_storage / "p2.json" if self.model_storage else None
        return [
            LinearCostModel(self, self.op_costs, self.knob_speedup),
            QoSModelP1(
                self,
                lambda conf: self._run_on_knobs(conf, False)[0],
                self._compute_accuracy,
                p1_storage,
            ),
            QoSModelP2(self, p2_storage),
        ]

    def _compute_accuracy(self, output_tensor: torch.Tensor) -> float:
        return accuracy(output_tensor, self.tune_labels)

    def _invoke_binary(self):
        import atexit
        import subprocess

        make_fifo(self.fifo_r_file)
        make_fifo(self.fifo_w_file)
        null_file = open(os.devnull, "wb")
        self.process = subprocess.Popen(
            [self.binary_path], stdout=null_file, cwd=self.base_dir
        )
        atexit.register(self._stop_binary)

    def _run_on_knobs(
        self, with_approxes: KnobsT, is_test: bool
    ) -> Tuple[torch.Tensor, float]:
        self._check_running()
        conf = self.add_baseline_to_knobs(with_approxes)
        with self.conf_file.open("w") as f:
            f.write(self.knob_exporter.knobs_to_str(conf))
        with self.fifo_w_file.open("w") as f:
            f.write("test" if is_test else "tune")
        ret = read_till_end(self.fifo_r_file)
        self._check_running()
        ret_tensor = parse_hpvm_tensor(ret)
        with self.qos_file.open() as f:
            qos = float(f.read())
        # Just in case of duplicate read, remove final_accuracy file
        self.qos_file.unlink()
        return ret_tensor, qos

    def _stop_binary(self):
        if self.process.poll() is not None:
            return
        self.process.terminate()

    def _check_running(self):
        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Subprocess has unexpectedly exited with code {return_code}"
            )

    @staticmethod
    def _read_metadata(metadata_path: PathLike):
        metadata_file = Path(metadata_path)
        with metadata_file.open() as f:
            metadata = json.load(f)
        op_costs = metadata["op_cost"]
        op_knobs = metadata["op_knobs"]
        knobs = metadata["knobs"]
        # Check sanity
        if set(op_costs.keys()) != set(op_knobs.keys()):
            raise ValueError(
                "Operators listed in layer_cost and knobs_of_layer mismatch"
            )
        knobs_used = set().union(*[set(knobs) for knobs in op_knobs.values()])
        knobs_defined = set(knobs.keys())
        undefined = knobs_used - knobs_defined
        if undefined:
            raise ValueError(
                f"These knobs used in knobs_of_layer are undefined: {undefined}"
            )
        # Create actual knob object from knob names
        knob_objs = {}
        knob_speedup = {}
        for knob_name, knob_args in knobs.items():
            knob_speedup[knob_name] = knob_args.pop("speedup")
            knob_objs[knob_name] = ApproxKnob(knob_name, **knob_args)
        op_knobs = {op: [knob_objs[k] for k in knobs] for op, knobs in op_knobs.items()}
        # Process other fields in metadata
        tune_labels_file = Path(metadata["tune_labels_path"])
        tune_labels = torch.from_numpy(np.fromfile(tune_labels_file, dtype=np.int32))
        conf_file = Path(metadata["conf_path"])
        # -- Our "w" file is the binary's "r" file, vice versa
        fifo_r_file = Path(metadata["fifo_path_w"])
        fifo_w_file = Path(metadata["fifo_path_r"])
        return (
            op_costs,
            op_knobs,
            knob_speedup,
            tune_labels,
            conf_file,
            fifo_r_file,
            fifo_w_file,
        )


def parse_hpvm_tensor(buffer: bytes) -> torch.Tensor:
    offset = 0
    batches = []
    while offset < len(buffer):
        ndims = np.frombuffer(buffer, dtype=np.uint64, offset=offset, count=1)
        ndims = int(ndims[0])
        offset += 1 * 8
        dims = np.frombuffer(buffer, dtype=np.uint64, offset=offset, count=ndims)
        nelem = int(np.prod(dims))
        offset += ndims * 8
        data = np.frombuffer(buffer, dtype=np.float32, offset=offset, count=nelem)
        offset += nelem * 4
        batches.append(data.reshape(*dims))
    if not batches:
        raise ValueError("No tensor returned from HPVM binary")
    batches = np.concatenate(batches, axis=0)
    return torch.from_numpy(batches).squeeze(-1).squeeze(-1)


def read_till_end(filepath: PathLike) -> bytes:
    data = []
    with open(filepath, "rb") as fifo:
        while True:
            part = fifo.read()
            if len(part) == 0:
                break
            data.append(part)
    return b"".join(data)


def make_fifo(path: Path):
    if path.exists():
        path.unlink()
    os.mkfifo(path)


def invert_knob_name_to_range(knob_name_to_range: Dict[str, range]):
    ret = {}
    for k, range_ in knob_name_to_range.items():
        for idx in range_:
            ret[str(idx)] = k
    return ret


class HPVMConfigBuilder:
    max_merge_chain = [
        ["convolution", "linear"],
        ["add"],
        ["tanh", "relu"],
        ["maxpool"],
    ]

    op_to_op = {
        "convolution": "conv",
        "maxpool": "pool_max",
        "linear": "mul",
        "avgpool": "pool_mean",
        "depthwise_convolution": "group_conv",
    }

    knob_name_to_range = {
        "fp32": range(11, 12),
        "fp16": range(12, 13),
        "perf": range(121, 158 + 1),
        "samp": range(231, 239 + 1),
        "perf_fp16": range(151, 168 + 1),
        "samp_fp16": range(261, 269 + 1),
    }

    knob_to_knob = invert_knob_name_to_range(knob_name_to_range)

    def __init__(self, ops: List[str]) -> None:
        self.ops = ops
        self.types = self._parse_ops(ops)
        self.merged_to_original = self._find_merge_chains(self.types)

    def knobs_to_str(self, knobs: KnobsT) -> str:
        dummy_conf = ValConfig(1.0, 1.0, knobs, 1.0, 1.0)
        return self.configs_to_str([dummy_conf], 1.0)

    def configs_to_str(self, configs: Sequence[ValConfig], base_test_qos: float) -> str:
        body = "\n".join(
            self.config_to_str(config, base_test_qos, i)
            for i, config in enumerate(configs)
        )
        return f"0.0\n{body}"

    def config_to_str(
        self, config: ValConfig, base_test_qos: float, conf_idx: int = 1
    ) -> str:
        def print_line(line_index: int, op_indices):
            ops = " ".join(self._print_op(config.knobs, idx) for idx in op_indices)
            return f"{line_index} gpu {ops}"

        if len(config.knobs) != len(self.ops):
            raise ValueError(f"Incorrect config length, expected {len(self.ops)}")
        test_qos = config.test_qos
        qos_loss = base_test_qos - config.test_qos
        conf_header = f"conf{conf_idx} {config.speedup} 1.0 {test_qos} {qos_loss}"
        body_lines = [
            print_line(line_idx, orig_indices)
            for line_idx, orig_indices in enumerate(self.merged_to_original, start=1)
        ]
        return "\n".join(["+++++", conf_header] + body_lines + ["-----"])

    def _print_op(self, knobs: KnobsT, op_index: int):
        ty = self.types[op_index]
        knob_value = knobs[self.ops[op_index]]
        out_knob_ty = self.knob_to_knob[knob_value]
        out_op_ty = self.op_to_op.get(ty, ty)
        return f"{out_op_ty} {out_knob_ty} {knob_value}"

    @staticmethod
    def _parse_ops(ops: List[str]) -> List[str]:
        types: List[str] = [None for _ in range(len(ops))]
        for k in ops:
            try:
                ty, idx_s = k.rsplit("_", 1)
                idx = int(idx_s)
            except ValueError as e:
                raise ValueError(
                    f"Operator name {k} not understood. Original parsing error:\n"
                    f"{e}"
                )
            types[idx] = ty
        if any(x is None for x in types):
            raise ValueError("Operator indice not consecutive")
        return types

    @classmethod
    def _find_merge_chains(cls, types: List[str]):
        mm = cls.max_merge_chain
        lhs, rhs = 0, 0  # rhs >= lhs
        merged_to_original = []
        while lhs < len(types):
            widx = 0
            while widx < len(mm) and rhs < len(types) and types[rhs] in mm[widx]:
                rhs += 1
                widx = rhs - lhs
            if rhs == lhs:
                rhs = lhs + 1  # At least take 1 operator
            merged_to_original.append(range(lhs, rhs))
            lhs = rhs
        return merged_to_original
