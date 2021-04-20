import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, IO, List, NamedTuple, Optional, Sequence, Tuple, Union

import onnx
import torch
from onnx import version_converter
from torch.nn import Module
from torch.utils.data import Dataset

from .codegen_hpvm import HpvmCodeGen, PathLike
from .codegen_tensor import TensorCodeGen
from .graph_builder import DFG


class BinDataset(NamedTuple):
    input_file: PathLike
    labels_file: PathLike
    shape: Sequence[int]


# Path to ONNX model, loaded ONNX model, or PyTorch Module
ModelTy = Union[PathLike, onnx.ModelProto, Module]
# Path to a pair of HPVM bin format file (input, labels), or PyTorch Dataset
DatasetTy = Union[BinDataset, Dataset]

def_approx_knobs_file = Path(__file__).parent / "approxknobs.json"


class ModelExporter:
    tuneset_name = "tune_input.bin", "tune_labels.bin"
    testset_name = "test_input.bin", "test_labels.bin"
    weight_dir_name = "weights"
    source_file_name = "hpvm_c.cpp"
    metadata_file_name = "ops.json"
    config_file_name = "tuner_confs.txt"
    fifo_file_name_r = "hpvm_fifo_r"
    fifo_file_name_w = "hpvm_fifo_w"

    def __init__(
        self,
        model: ModelTy,
        tune_dataset: DatasetTy,
        test_dataset: DatasetTy,
        output_dir: PathLike,
        target: str = "hpvm_tensor",
        opset: Optional[int] = None,
        config_file: PathLike = None,
    ):
        self.tune_dataset, self.test_dataset = tune_dataset, test_dataset
        self.dataset_shape = self._check_datasets(tune_dataset, test_dataset)
        self.dataset_size = self.dataset_shape[0]
        onnx_model = self._load_model(model, self.dataset_shape, opset)
        self.dfg = DFG(onnx_model.graph)

        output_dir = Path(output_dir).absolute()
        os.makedirs(output_dir, exist_ok=False)  # Will throw if already existss
        self.weight_dir = output_dir / self.weight_dir_name
        self.weight_dir.mkdir(exist_ok=True)
        self.codefile = output_dir / self.source_file_name
        self.metafile = output_dir / self.metadata_file_name

        args3 = self.dfg, self.weight_dir, self.dataset_size
        self.compile_args = None
        self.path_params = {}
        if target == "hpvm_tensor":
            if config_file is None:
                raise ValueError(
                    f"Config file must be given and exist under hpvm_tensor mode"
                )
            self.path_params = {"config_file": str(config_file)}
            self.compile_args = ["-t", "tensor", "--conf-file", str(config_file)]
            self.codegen = HpvmCodeGen(*args3, "tensor", None)
        elif target == "hpvm_tensor_inspect":
            if config_file is None:
                config_file = output_dir / self.config_file_name
            else:
                config_file = Path(config_file).absolute()
            self.path_params = {
                "tune_labels_path": (self.weight_dir / self.tuneset_name[1]).as_posix(),
                "conf_path": config_file.as_posix(),
                "fifo_path_r": (output_dir / self.fifo_file_name_r).as_posix(),
                "fifo_path_w": (output_dir / self.fifo_file_name_w).as_posix(),
            }
            self.compile_args = ["-t", "tensor", "--conf-file", str(config_file)]
            self.codegen = HpvmCodeGen(*args3, "tensor", self.path_params)
        elif target == "hpvm_cudnn":
            self.compile_target = "cudnn"
            self.compile_args = ["-t", "cudnn"]
            self.codegen = HpvmCodeGen(*args3, "cudnn", None)
        elif target == "tensor":
            self.codegen = TensorCodeGen(*args3)
        else:
            raise ValueError(f"Target {target} not recognized")

    def export_source_code(self, output: PathLike, batch_size: Optional[int] = None):
        self.codegen.compile(output, batch_size)
        return self

    def export_weights(self):
        self.dfg.dump_weights(self.weight_dir)
        return self

    def export_datasets(self):
        input_, labels = self.tuneset_name
        self._dump_dataset(
            self.tune_dataset, self.weight_dir / input_, self.weight_dir / labels
        )
        input_, labels = self.testset_name
        self._dump_dataset(
            self.test_dataset, self.weight_dir / input_, self.weight_dir / labels
        )
        return self

    def export_metadata(
        self, output: PathLike, approx_knobs_file: PathLike = def_approx_knobs_file
    ):
        import json
        from collections import defaultdict

        with Path(approx_knobs_file).open() as f:
            knobs = json.load(f)  # knob name to knob attrs dict
        # Organize knobs into defaults and the ones for certain types
        ty_knobs: Dict[str, str] = defaultdict(list)
        default_knobs: List[str] = []
        for name, attrs in knobs.items():
            applies_to = attrs.pop("applies_to")
            if applies_to is None:
                default_knobs.append(name)
                continue
            for ty in applies_to:
                ty_knobs[ty].append(name)

        # Enumerate operators and find knobs for each
        idx = 0
        op_cost: Dict[str, int] = {}
        op_knobs: Dict[str, List[str]] = {}
        for node in self.dfg.traverse_order:
            if not node.hpvm_op_type:
                continue
            hpvm_op_name = f"{node.hpvm_op_type}_{idx}"
            type_knobs = ty_knobs.get(node.hpvm_op_type, [])
            op_knobs[hpvm_op_name] = type_knobs + default_knobs
            op_cost[hpvm_op_name] = int(node.get_flops())  # May get np.int64
            idx += 1

        # Write out
        with Path(output).open("w") as f:
            json.dump(
                {
                    "op_cost": op_cost,
                    "op_knobs": op_knobs,
                    "knobs": knobs,
                    **self.path_params,
                },
                f,
                indent=2,
            )
        return self

    def compile(self, output_binary: PathLike, working_dir: Optional[PathLike] = None):
        from subprocess import run

        args = [
            "hpvm-clang",
            "-O3",
            str(self.codefile),
            str(output_binary),
            *self.compile_args,
        ]
        if working_dir is not None:
            args.extend(["-d", str(working_dir)])
        run(args, check=True)
        return self

    def generate(
        self, output_code_file: PathLike = None, batch_size: Optional[int] = None
    ):
        self.codefile = (
            self.codefile if output_code_file is None else Path(output_code_file)
        )
        self.export_source_code(self.codefile, batch_size)
        self.export_metadata(self.metafile)
        self.export_weights()
        self.export_datasets()
        return self

    @staticmethod
    def _dump_dataset(dataset: DatasetTy, input_filename: Path, labels_filename: Path):
        import numpy as np
        from torch.utils.data import DataLoader

        def link_from_to(from_: PathLike, to: PathLike):
            from_, to = Path(from_), Path(to)
            if from_.exists():
                from_.unlink()
            from_.symlink_to(to.absolute())

        if isinstance(dataset, BinDataset):
            link_from_to(input_filename, dataset.input_file)
            link_from_to(labels_filename, dataset.labels_file)
            return
        inputs, labels = zip(*iter(DataLoader(dataset)))
        inputs = np.stack(inputs, axis=0)
        labels = np.stack(labels, axis=0)
        inputs.tofile(input_filename)
        inputs.tofile(labels_filename)

    @classmethod
    def _check_datasets(
        cls, tune_dataset: DatasetTy, test_dataset: DatasetTy
    ) -> Tuple[int, int, int, int]:
        tune_shape = cls._check_dataset_get_shape(tune_dataset)
        test_shape = cls._check_dataset_get_shape(test_dataset)
        if tune_shape != test_shape:
            raise ValueError(
                f"Size of tune and test dataset must match (got {tune_shape} and {test_shape})"
            )
        return tuple(tune_shape)

    @staticmethod
    def _check_dataset_get_shape(dataset: DatasetTy) -> Sequence[int]:
        import numpy as np

        if isinstance(dataset, Dataset):
            size = len(dataset)
            sample = dataset[0]
            if (
                not isinstance(sample, (np.ndarray, torch.Tensor))
                or len(sample.shape) != 4
            ):
                raise ValueError(
                    "Dataset must be a 4D tensor due to backend limitation"
                )
            return [size, *sample.shape]
        if not isinstance(dataset, BinDataset):
            raise TypeError("Only BinDataset or PyTorch Dataset are supported")
        input_file = Path(dataset.input_file)
        labels_file = Path(dataset.labels_file)
        if not input_file.is_file():
            raise FileNotFoundError(f"Input file {input_file}")
        if not labels_file.is_file():
            raise FileNotFoundError(f"Labels file {input_file}")
        if len(dataset.shape) != 4:
            raise ValueError("Dataset must be a 4D tensor due to backend limitation")
        float_size = np.dtype(np.float32).itemsize
        expected_input_size = np.array(dataset.shape).prod() * float_size
        int32_size = np.dtype(np.int32).itemsize
        expected_labels_size = dataset.shape[0] * int32_size
        input_size = input_file.stat().st_size
        labels_size = labels_file.stat().st_size
        if input_size != expected_input_size:
            raise RuntimeError(
                f"Input file {input_file} should have size {expected_input_size} "
                f"(shape {dataset.shape}), but got {input_size}"
            )
        if labels_size != expected_labels_size:
            raise RuntimeError(
                f"Labels file {labels_file} should have size {expected_labels_size} "
                f"(dataset length {dataset.shape[0]}), but got {labels_size}"
            )
        return dataset.shape

    @staticmethod
    def _load_model(
        model: ModelTy, dataset_shape: Sequence[int], opset: Optional[int]
    ) -> onnx.ModelProto:
        from onnxsim import simplify

        if isinstance(model, Module):
            # Export to ONNX and load back.
            sample_input_shape = 1, *dataset_shape[1:]
            sample_input = torch.rand(sample_input_shape)
            with NamedTemporaryFile("w+b") as tmp:
                torch_to_onnx(model, (sample_input,), tmp)
                tmp.seek(0)
                onnx_model = onnx.load_model(tmp)
        elif isinstance(model, onnx.ModelProto):
            onnx_model = model
        else:
            raise ValueError(f"Cannot accept model of type {type(model)}")
        if opset is not None:
            onnx_model = check_onnx_version(onnx_model, opset)
        onnx_model, check = simplify(
            onnx_model, skip_fuse_bn=True, skipped_optimizers=["fuse_bn_into_conv"]
        )
        assert check, "Simplified ONNX model could not be validated"
        return onnx.shape_inference.infer_shapes(onnx_model)


def check_onnx_version(model, new_version):
    try:
        opset = model.opset_import[0].version if model.opset_import else 1
    except AttributeError:
        opset = 1  # default opset version set to 1 if not specified
    if opset != new_version:
        try:
            converted_model = version_converter.convert_version(model, new_version)
            return converted_model
        except RuntimeError as e:
            raise RuntimeError(
                f"Current version {opset} of ONNX model not supported!\n"
                f"Conversion failed with message below: \n{e}"
            )
    return model


def torch_to_onnx(
    module_cpu: Module,
    model_args_cpu: tuple,
    output_obj: Union[IO, PathLike],
    opset_version: int = 10,
):
    from torch.onnx import export

    # Export the model (must be on CPU, some model only supports this)
    export(
        module_cpu.eval(),
        model_args_cpu,
        output_obj,
        export_params=True,  # store the trained parameter weights inside the model file
        do_constant_folding=False,
        opset_version=opset_version,  # the ONNX version to export the model to
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
    )


def get_full_typename(o: object) -> str:
    return o.__module__ + "." + o.__class__.__qualname__
