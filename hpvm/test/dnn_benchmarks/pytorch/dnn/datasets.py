import logging
from pathlib import Path
from typing import Iterator, Tuple, Union

import numpy as np
import torch
from torch.utils.data.dataset import Dataset

RetT = Tuple[torch.Tensor, torch.Tensor]
msg_logger = logging.getLogger(__name__)

PathLike = Union[Path, str]


class SingleFileDataset(Dataset):
    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor):
        self.inputs, self.outputs = inputs, outputs

    @classmethod
    def from_file(cls, *args, **kwargs):
        pass

    @property
    def sample_input(self):
        inputs, outputs = next(iter(self))
        return inputs

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx) -> RetT:
        return self.inputs[idx], self.outputs[idx]

    def __iter__(self) -> Iterator[RetT]:
        for i in range(len(self)):
            yield self[i]


class DNNDataset(SingleFileDataset):
    image_shape = None
    label_ty = np.int32

    @classmethod
    def from_file(
        cls,
        input_file: PathLike,
        labels_file: PathLike,
        count: int = -1,
        offset: int = 0,
    ):
        # NOTE: assuming (N, *) ordering of inputs (such as NCHW, NHWC)
        channel_size = np.prod(np.array(cls.image_shape))
        inputs_count_byte = -1 if count == -1 else count * channel_size
        inputs = read_tensor_from_file(
            input_file,
            -1,
            *cls.image_shape,
            count=inputs_count_byte,
            offset=offset * channel_size,
        )
        labels = read_tensor_from_file(
            labels_file,
            -1,
            read_ty=cls.label_ty,
            cast_ty=np.long,
            count=count,
            offset=offset,
        )
        if inputs.shape[0] != labels.shape[0]:
            raise ValueError("Input and output have different number of data points")
        msg_logger.info(f"%d entries loaded from dataset.", inputs.shape[0])
        return cls(inputs, labels)


class MNIST(DNNDataset):
    image_shape = 1, 28, 28


class CIFAR(DNNDataset):
    image_shape = 3, 32, 32


class ImageNet(DNNDataset):
    image_shape = 3, 224, 224


def read_tensor_from_file(
    filename: Union[str, Path],
    *shape: int,
    read_ty=np.float32,
    cast_ty=np.float32,
    count: int = -1,
    offset: int = 0,
) -> torch.Tensor:
    offset = offset * read_ty().itemsize
    mmap = np.memmap(filename, dtype=read_ty, mode="r", offset=offset)
    n_entries = min(mmap.shape[0], count) if count != -1 else mmap.shape[0]
    np_array = mmap[:n_entries].reshape(shape).astype(cast_ty)
    return torch.from_numpy(np_array).clone()
