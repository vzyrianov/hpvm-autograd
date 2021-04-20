from typing import Optional, Union

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data._utils.fetch import _BaseDatasetFetcher
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter


def infer_net_device(net: Module):
    """Guess the device `net` is on.

    This assumes its all parts are on the same device, and takes the device of any parameter.
    This function does not check the device of buffers, etc. in `net`."""
    devices = set(pm.device for pm in net.parameters())
    if len(devices) == 0:
        raise RuntimeError("Cannot infer device for net with no parameters")
    if len(devices) > 1:
        raise RuntimeError("Parts of the network are on different devices")
    (device,) = devices
    return device


def move_to_device_recursively(data: object, device: Union[torch.device, str]):
    """Move all Tensors in `data` recursively to `device`."""
    if isinstance(data, Tensor):
        return data.to(device)
    if not hasattr(data, "__dict__"):
        if isinstance(data, list):
            return [move_to_device_recursively(x, device) for x in data]
        elif isinstance(data, tuple):
            return tuple([move_to_device_recursively(x, device) for x in data])
        else:
            raise RuntimeError(f"Don't know how to manipulate {type(data)}")
    for key, value in data.__dict__.items():
        data.__dict__[key] = move_to_device_recursively(value, device)
    return data


def split_dataset(dataset: Dataset, split_at: int):
    return (
        Subset(dataset, torch.arange(0, split_at)),
        Subset(dataset, torch.arange(split_at, len(dataset))),
    )


class BatchedDataLoader(DataLoader):
    """Faster data loader for datasets that supports batch indexing.

    Some datasets load the whole Tensor into memory and can be indexed by a batch of indices,
    instead of indexed one by one and stacking the data together (which is what DataLoader does).
    `BatchedDataLoader` instead uses `_BatchedMapDatasetFetcher` to batch index the dataset,
    removing some overhead.
    """

    def __init__(self, dataset: Dataset, batch_size: Optional[int], *args, **kwargs):
        super().__init__(dataset, batch_size=batch_size, *args, **kwargs)
        try:
            next(iter(self))
            self.support_batch = True
        except (KeyError, ValueError, RuntimeError):
            self.support_batch = False

    def __iter__(self):
        if self.num_workers == 0 and self.support_batch:
            dl_iter = _SingleProcessDataLoaderIter(self)
            dl_iter._dataset_fetcher = _BatchedMapDatasetFetcher(
                self.dataset, self._auto_collation, self.collate_fn, self.drop_last
            )
            return dl_iter
        return super(BatchedDataLoader, self).__iter__()


class _BatchedMapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        return self.dataset[possibly_batched_index]
