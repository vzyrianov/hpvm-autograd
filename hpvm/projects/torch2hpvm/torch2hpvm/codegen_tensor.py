from pathlib import Path
from typing import Dict, List, Optional

import jinja2

from .codegen_hpvm import CodeGen, PathLike
from .graph_ir import DFGNode, TensorNode

TEMPLATE_FILE = "template_tensor.cpp.in"
loader = jinja2.FileSystemLoader(searchpath=Path(__file__).parent)
template_env = jinja2.Environment(loader=loader, trim_blocks=True)
template = template_env.get_template(TEMPLATE_FILE)


class TensorCodeGen(CodeGen):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.variables: Dict[DFGNode, str] = {
            n: self.make_c_identifier(n.name) for n in self.root_args
        }

    ################################################
    # CodeGen functions
    ################################################

    def emit_graph(self) -> List[dict]:
        graph_code = []
        for node in self.dfg.traverse_order:
            if isinstance(node, TensorNode):
                continue
            inputs = self.dfg.node_args(node)
            func_name, extra_args = node.codegen()
            if func_name == "":  # No code generation
                # Node must have single input, we equate the output to
                # the input and skip code generation.
                assert len(inputs) == 1
                self.variables[node] = self.variables[inputs[0]]
                continue
            varname = f"var_{self._inc_var_count()}"
            self.variables[node] = varname
            input_args = [self.variables[n] for n in inputs] + extra_args
            graph_code.append(
                {"output": varname, "inputs": input_args, "function": func_name}
            )
        return graph_code

    ################################################
    # Compile is a top level function to compile an onnx model into C/C++
    # program with HPVM Tensor Runtime
    ################################################

    def compile(self, output: PathLike, batch_size: Optional[int] = None):
        graph_code = self.emit_graph()
        output_arg = self.variables[self.dfg.output]
        with Path(output).open("w") as f:
            f.write(
                template.render(
                    input=self.input_name,
                    input_shape=self.input_shape,
                    output=output_arg,
                    graph_code=graph_code,
                    weights=self.emit_weights(self.weights),
                    prefix=self.prefix,
                )
            )
