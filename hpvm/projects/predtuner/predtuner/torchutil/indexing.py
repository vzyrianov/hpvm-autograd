"""Tools for indexing into an nn.Module with layer name (str) or index (int)."""
from typing import Callable, Dict, Iterator, Optional, Set, Tuple, Union

from torch.nn import Module, Sequential

ModulePredT = Callable[[Module], bool]


class ModuleIndexer:
    r"""Allows indexing into an nn.Module with index (int) to get layers.
    Supports read and modification, just like a dictionary.

    Parameters
    ----------
    module: Module
        The PyTorch Module to be indexed.
    include_module: Callable[[Module], bool] = None
        A predicate that decides which layers to include in the index. For example,
        `lambda layer: isinstance(layer, Conv2d)` tells `ModuleIndexer` to only include `Conv2d`
        layers.
        If not given, by default `ModuleIndexer` will recursively walk down `module` like a tree
        to include all internal and leaf nodes (layers), except for layers that `expand_module`
        forbids recursing into.
    expand_module: Callable[[Module], bool] = None
        A predicate that decides which layers to recurse down. If `expand_module` returns `False`,
        layer is kept as a whole and may be included if `include_module` allows.

    Attributes
    ----------
    module: Module
        Equal to parameter `module`.
    index_to_module: List[Module]
        Stores the layers in order so that a layer at `index_to_module[i]` has the index `i`.
    layer_parent: Dict[Module, Tuple[Module, str]]
        Maps each layer to its parent and its name in the parent layer. Contains the same layers
        as in `index_to_module` except `module` which has no parent.
    """

    def __init__(
        self,
        module: Module,
        include_module: Optional[ModulePredT] = None,
        expand_module: Optional[ModulePredT] = None,
    ):
        self.module = module
        self.index_to_module = []
        self.module_to_name = {}
        self.name_to_index = {}
        # By default, don't include container layer, and don't include (empty) Sequential
        has_children = lambda m: bool(list(m.children()))
        default_inclusion = lambda m: not has_children(m) and not isinstance(
            m, Sequential
        )
        # No need for "default expansion" because whatever is not included will be walked into.
        self._rec_expand_module(
            module, "", include_module or default_inclusion, expand_module
        )
        self.layer_parent = self._find_layers_parent_info(module, set(self.all_modules))

    def _rec_expand_module(
        self,
        module: Module,
        name_prefix: str,
        include_module: ModulePredT,
        expand_module: Optional[ModulePredT],
    ):
        """Recursively expands into module and builds the index."""
        for name, submodule in module.named_children():
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            included = include_module(submodule)
            if included:
                self.index_to_module.append(submodule)
                self.module_to_name[submodule] = full_name
                self.name_to_index[full_name] = len(self.index_to_module) - 1
            required_expansion = expand_module and expand_module(submodule)
            default_expansion = not included
            if default_expansion or required_expansion:
                self._rec_expand_module(
                    submodule, full_name, include_module, expand_module
                )

    @staticmethod
    def _find_layers_parent_info(net: Module, layers: Set[Module]):
        """Find parent info for each child layer in `net`, ignoring those not in `layers`."""
        ret = {}
        for name, submodule in net.named_children():
            if submodule in layers:
                ret[submodule] = net, name
            ret = {**ret, **ModuleIndexer._find_layers_parent_info(submodule, layers)}
        return ret

    @property
    def all_modules(self) -> Iterator[Module]:
        return iter(self.index_to_module)

    @property
    def name_to_module(self) -> Dict[str, Module]:
        return {
            name: self.index_to_module[index]
            for name, index in self.name_to_index.items()
        }

    def find_by_module(self, module: Module) -> Optional[Tuple[str, int]]:
        """Get name and index from module."""
        name = self.module_to_name.get(module, None)
        if name is None:
            return None
        index = self.name_to_index[name]
        return name, index

    def __getitem__(self, item: Union[int, str]) -> Module:
        """Get module from index."""
        if isinstance(item, int):
            return self.index_to_module[item]
        elif isinstance(item, str):
            return self[self.name_to_index[item]]
        raise KeyError(f"Key type {item.__class__} not understood")

    def __setitem__(self, key: Union[int, str], value: Module):
        """Swap in the layer at index `key` to be `value`.

        The parent of the old layer at `key` is also updated with the new layer, so that `self.module`
        has the old layer replaced with new.
        """
        if isinstance(key, str):
            key = self.name_to_index[key]
        old = self.index_to_module[key]
        if value != old:
            self.index_to_module[key] = value
            self.module_to_name[value] = self.module_to_name.pop(old)
            parent, name = self.layer_parent[old]
            self.layer_parent[value] = parent, name
            self.layer_parent.pop(old)
            parent.__setattr__(name, value)

    def __iter__(self) -> Iterator[Module]:
        return self.all_modules

    def __len__(self):
        """Number of indexed layers."""
        return len(self.index_to_module)
