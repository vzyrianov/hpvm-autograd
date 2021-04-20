from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import jinja2

from .graph_builder import DFG
from .graph_ir import DFGNode, TensorNode, WeightTensor

PLAIN_TEMPLATE_FILE = "template_hpvm.cpp.in"
INSPECT_TEMPLATE_FILE = "template_hpvm_inspect.cpp.in"
loader = jinja2.FileSystemLoader(searchpath=Path(__file__).parent)
template_env = jinja2.Environment(loader=loader, trim_blocks=True)

PathLike = Union[str, Path]


class CodeGen:
    def __init__(self, dfg: DFG, prefix: PathLike, input_size: int):
        self.dfg = dfg
        self.var_count = 0
        self.prefix = Path(prefix)
        # Some reasoning of input information
        assert len(self.dfg.inputs) == 1
        input_tensor = self.dfg.inputs[0]
        self.input_name = input_tensor.name
        self.input_shape = input_tensor.shape[1:]
        self.input_size = input_size
        # self.variables is a "node to our name" map
        # Each value is (varname, bool) and the bool indicates
        # "is root node input" or not.
        self.root_args = sorted(
            [n for n in dfg.traverse_order if isinstance(n, TensorNode)],
            key=lambda n: n.name,
        )
        self.weights = [n for n in self.root_args if isinstance(n, WeightTensor)]
        self.variables: Dict[DFGNode, Tuple[Any, bool]] = {
            f_name: (index, True) for index, f_name in enumerate(self.root_args)
        }

    ################################################
    # Aux functions
    ################################################

    def _inc_var_count(self) -> int:
        var_n = self.var_count
        self.var_count += 1
        return var_n

    @classmethod
    def emit_weights(cls, weights: List[WeightTensor]) -> List[dict]:
        ret = []
        for weight in weights:
            name = cls.make_c_identifier(weight.name)
            file_path = f"{weight.new_name}_path.bin"
            ret.append(
                {"name": name, "shape": weight.output_shape, "filename": file_path}
            )
        return ret

    @staticmethod
    def make_c_identifier(name: str) -> str:
        name = name.replace(".", "_")
        if name[0].isnumeric():
            name = "_" + name
        return name


class HpvmCodeGen(CodeGen):
    # Variable indicator is always int for hpvm gen
    variables: Dict[DFGNode, Tuple[int, bool]]

    def __init__(
        self,
        dfg: DFG,
        prefix: PathLike,
        input_size: int,
        target: str,
        inspectable: Optional[dict],
    ):
        super().__init__(dfg, prefix, input_size)
        if target not in ("tensor", "cudnn"):
            raise ValueError(f"Unsupported target {target}")
        self.target = target
        self.template = template_env.get_template(
            PLAIN_TEMPLATE_FILE if inspectable is None else INSPECT_TEMPLATE_FILE
        )
        self.inspect_vars = inspectable or {}

    def _emit_hpvm_node_edges(self, input_vars: List[DFGNode]) -> List[dict]:
        ret = []
        it = 0
        for node in input_vars:
            hpvm_var_idx, is_root_input = self.variables[node]
            if is_root_input:
                assert isinstance(hpvm_var_idx, int)
                ret.append(
                    {"is_bindin": True, "input_idx": hpvm_var_idx, "edge_idx": it}
                )
            else:
                ret.append(
                    {"is_bindin": False, "input_node": hpvm_var_idx, "edge_idx": it}
                )
            it += 1
        return ret

    def emit_hpvm_node_structures(self) -> List[dict]:
        node_envs = []
        for node in self.dfg.traverse_order:
            if isinstance(node, TensorNode):
                continue
            inputs = self.dfg.node_args(node)
            func_name, extra_args = node.hpvm_codegen()
            if func_name == "":  # No code generation
                # Node must have single input, we equate the output to
                # the input and skip code generation.
                assert len(inputs) == 1
                self.variables[node] = self.variables[inputs[0]]
                continue
            var_idx = self._inc_var_count()
            self.variables[node] = var_idx, False  # not root-node arg
            node_envs.append(
                {
                    "idx": var_idx,
                    "input_size": len(inputs),
                    "edges": self._emit_hpvm_node_edges(inputs),
                    "call_name": func_name,
                    "call_args": extra_args,
                }
            )
        return node_envs

    def emit_root_io(self) -> Tuple[List[str], int]:
        input_args = [
            self.make_c_identifier(node.name)
            for node, (_, is_root) in self.variables.items()
            if is_root
        ]
        output_var_idx = self.variables[self.dfg.output][0]
        return input_args, output_var_idx

    def compile(self, output: PathLike, batch_size: Optional[int] = None) -> None:
        nodes = self.emit_hpvm_node_structures()
        inputs, output_var_idx = self.emit_root_io()
        weights = self.emit_weights(self.weights)
        with Path(output).open("w") as f:
            f.write(
                self.template.render(
                    nodes=nodes,
                    input_name=self.input_name,
                    input_size=self.input_size,
                    batch_size=batch_size or self.input_size,
                    input_shape=self.input_shape,
                    root_inputs=inputs,
                    root_output_idx=output_var_idx,
                    weights=weights,
                    prefix=self.prefix,
                    target=self.target,
                    **self.inspect_vars
                )
            )
