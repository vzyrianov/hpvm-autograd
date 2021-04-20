from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import networkx as nx
import onnx

from . import graph_ir as g
from .onnx_attr import get_node_shape, node_attr_to_dict, node_to_shape

PathLike = Union[str, Path]
GraphT = onnx.GraphProto
NodeT = onnx.NodeProto
NodeT.__hash__ = lambda self: id(self)
NodeT.__repr__ = NodeT.__str__ = lambda self: self.name


class MarkedSubGraph:
    """A subgraph with information on how it should replace a node in a super graph.

    subgraph: a nx.DiGraph subgraph
    entry_edges: a list of edges from nodes "outside" to nodes in self.subgraph
    exit: the exit node of the subgraph.
        When this subgraph replaces a node `n`, self.exit will be connected to
        whateven `n` is connected to.
    """

    def __init__(self, subgraph: nx.DiGraph, entry_edges, exit) -> None:
        assert all(to in subgraph for _, to, _ in entry_edges)
        assert exit in subgraph
        self.subgraph, self.exit = subgraph, exit
        self.entry_edges = [(f, t, {"index": i}) for f, t, i in entry_edges]

    @classmethod
    def idiomatic_1to2(cls, node1, node2, predecessors):
        """Create an idiomatic replacement as follow:

        node(arg1, arg2, arg3) -> node2(node1(arg1, arg2), arg3)"""
        p0, p1, p2 = predecessors
        graph = nx.DiGraph()
        graph.add_edge(node1, node2, index=0)
        return cls(graph, [(p0, node1, 0), (p1, node1, 1), (p2, node2, 1)], node2)


EmitNodeT = Union[MarkedSubGraph, g.DFGNode]


class DFG(object):
    """ONNX model translated into DFG with `DFGNode`s.

    This class has a DFG, input/output information, and a clear traverse order
    (think dominant tree), and is easier for CodeGen classes to work with."""

    def __init__(self, graph: GraphT):
        self._check_model(graph)
        self._var_count = 0
        # Build explicit DFG with ONNX nodes
        onnx_graph = self._build_onnx_dfg(graph)
        # Convert ONNX dfg into DFGNode DFG
        self.graph = self._build_dfg(onnx_graph)
        # Find out input nodes and output node (unique)
        # removing dead nodes along the way if any
        self.inputs, self.output = self._dce_get_io_info()

    ################ Interfaces:

    @property
    def traverse_order(self) -> List[g.DFGNode]:
        """Get topological order of computational graph by use-def relation."""
        return list(nx.topological_sort(self.graph))

    def node_args(self, node: g.DFGNode):
        """Get input arguments of node."""
        sorted_edges = sorted(self.graph.in_edges(node, "index"), key=lambda p: p[2])
        return [e[0] for e in sorted_edges]

    def dump_weights(self, output_dir: PathLike) -> None:
        """Dump `WeightTensor`s into output_dir."""
        output_dir = Path(output_dir)
        for node in self.graph.nodes:
            if not isinstance(node, g.WeightTensor):
                continue
            node.dump_weight(output_dir / (node.new_name + "_path.bin"))

    ################ Internal methods (high-level):

    @staticmethod
    def _check_model(onnx_graph: GraphT):
        """Check model validaty and single output (which is our limitation)"""

        import warnings

        from onnx import checker, onnx_cpp2py_export

        # try use onnx's own model checker before converting any model
        try:
            checker.check_graph(onnx_graph)
        except onnx_cpp2py_export.checker.ValidationError as e:
            warnings.warn(str(e))
        if any(len(n.output) > 1 for n in onnx_graph.node):
            raise ValueError("All node must have single output")
        if len(onnx_graph.output) > 1:
            raise ValueError("Graph must have single output")

    @staticmethod
    def _build_onnx_dfg(graph: GraphT) -> nx.DiGraph:
        """Creates a DiGraph (by use-def relation) of onnx nodes from onnx GraphProto.
        DiGraph is easier to use as a graph compared to GraphProto where use-def is implicit."""

        ret_graph = nx.DiGraph()
        onnx_defs, onnx_uses = def_use(graph.node)
        node_shape = node_to_shape(graph)
        node_and_attr = [(n, {"shape": shape}) for n, shape in node_shape.items()]
        ret_graph.add_nodes_from(node_and_attr)
        tensors = extract_tensors_from_graph(graph)
        tensor_and_attr = [(t, {"shape": t.output_shape}) for t in tensors.values()]
        ret_graph.add_nodes_from(tensor_and_attr)
        for onnx_value_name, use_nodes in onnx_uses.items():
            def_node = onnx_defs.get(onnx_value_name)
            if def_node is None:
                def_node = tensors[onnx_value_name]
            for use_node, used_at_narg in use_nodes:
                ret_graph.add_edge(def_node, use_node, index=used_at_narg)
        return ret_graph

    def _build_dfg(self, onnx_graph: nx.DiGraph) -> nx.DiGraph:
        """Translate _build_onnx_dfg output into DFGNode DFG.
        
        First run some passes to process subgraphs that needs to be
        processed together, then each unprocessed node is generated into
        1 or more nodes."""

        # Gemm in tensor_runtime does reshape automatically
        # it also doesn't have a dedicated reshape operator
        onnx_graph = drop_reshape_before_gemm(onnx_graph)
        # For each onnx node, generate our nodes
        node_to_nodes, error_nodes = {}, []
        for onnx_node in nx.topological_sort(onnx_graph):
            our_nodes = self._emit_node(onnx_graph, onnx_node)
            if our_nodes is None:
                error_nodes.append(onnx_node)
            else:
                node_to_nodes[onnx_node] = our_nodes
        if error_nodes:
            error_repr = [f"{n.name}({n.op_type})" for n in error_nodes]
            if len(error_nodes) > 10:  # Magic number
                raise ValueError(f"Unsupported operators (first 10): {error_repr[:10]}")
            else:
                raise ValueError(f"Unsupported operators: {error_repr}")
        # Apply node_to_nodes replacement on onnx_graph to create a new DFG
        return build_graph_with_mapping(onnx_graph, node_to_nodes)

    def _dce_get_io_info(self):
        inputs = [n for n in self.graph if isinstance(n, g.InputTensor)]
        inputs_set = set(inputs)
        reachables = set()
        for component in nx.connected_components(self.graph.to_undirected()):
            # If any inputs goes into this subgraph, it's alive.
            if set(component).intersection(inputs_set):
                reachables.update(component)
        unreachables = set(self.graph) - reachables
        # Remove nodes unreachable from input
        self.graph.remove_nodes_from(unreachables)
        # Then outputs are nodes with out_degree = 0
        outputs = [n for n in self.graph if self.graph.out_degree[n] == 0]
        assert len(outputs) == 1
        return inputs, outputs[0]

    @staticmethod
    def _emit_node(in_graph: nx.DiGraph, node: NodeT) -> Optional[EmitNodeT]:
        output_shape = in_graph.nodes[node].get("shape")
        predec = sorted_inputs(in_graph, node)
        predec_shapes = [in_graph.nodes[n].get("shape") for n in predec]
        if isinstance(node, g.DFGNode):
            # Directly add node into return graph.
            return node
        attrs = node_attr_to_dict(node)
        attrs["input_shapes"] = predec_shapes
        attrs["output_shape"] = output_shape

        if node.op_type == "Conv":
            if not isinstance(predec[1], g.WeightTensor) or len(predec_shapes[1]) != 4:
                return None  # Only supports 2D conv with rhs being constant
            # Only pass in the first 2 arguments' shapes
            attrs["input_shapes"] = predec_shapes[:2]
            conv_node = g.Conv2DNode(node.name, **attrs)
            if len(predec) == 2:
                return conv_node
            # Split into conv followed by an addition
            bias_node = g.BiasAddNode(
                f"Bias_{node.name.split('_')[-1]}", [output_shape], output_shape
            )
            return MarkedSubGraph.idiomatic_1to2(conv_node, bias_node, predec)
        if node.op_type in ("MatMul", "Gemm"):
            attrs["input_shapes"] = predec_shapes[:2]
            mul_node = g.MatMulNode(node.name, **attrs)
            if node.op_type == "Gemm":
                mul_node.gemm_transpose(predec)
            if len(predec) == 2:
                return mul_node
            # Split into mul followed by an addition
            bias_node = g.BiasAddNode(
                f"Bias_{node.name.split('_')[-1]}", [output_shape], output_shape
            )
            return MarkedSubGraph.idiomatic_1to2(mul_node, bias_node, predec)
        if node.op_type == "GlobalAveragePool":
            input0_shape = in_graph.nodes[predec[0]]["shape"]
            _, _, h, w = input0_shape
            return g.AveragePool2DNode(
                node.name, predec_shapes, output_shape, [1, 1], (h, w), [0, 0, 0, 0]
            )
        one_to_one_nodes = {
            "MaxPool": g.MaxPool2DNode,
            "AveragePool": g.AveragePool2DNode,
            "Add": g.AddNode,
            "Softmax": g.SoftMaxNode,
            "Relu": g.ReluNode,
            "Tanh": g.TanhNode,
            "BatchNormalization": g.BatchNormalizationNode,
            "Pad": g.PadNode,
            "Identity": g.IdentityNode,
            "Flatten": g.FlattenNode,
        }
        if node.op_type not in one_to_one_nodes:
            return None
        try:
            return one_to_one_nodes[node.op_type](node.name, **attrs)
        except (TypeError, KeyError, ValueError, RuntimeError):
            node_class = one_to_one_nodes[node.op_type]
            raise ValueError(f"Node ({node_class}) creation failed")


def def_use(nodes: Iterable) -> Tuple[dict, dict]:
    """Computes def/use relation from a list of node.

    This method is duck-typed and operates on any node defining .input and .output.
    """
    defs, uses = {}, defaultdict(list)
    for n in nodes:
        for i, input_ in enumerate(n.input):
            uses[input_].append((n, i))
        for output in n.output:
            defs[output] = n
    return defs, uses


def drop_reshape_before_gemm(graph: nx.DiGraph) -> nx.DiGraph:
    """Look for a shape-gather-unsqueeze-concat-reshape chain and replace that with flatten."""
    for node in list(graph.nodes):
        if node.op_type != "Reshape":
            continue
        reshape_input, target_shape = sorted_inputs(graph, node)
        if not isinstance(target_shape, g.WeightTensor):  # Not constant shape, nope
            continue
        n_gemm = get_next_in_chain(graph, "Gemm", node)
        if n_gemm is None:
            continue
        # Must be an (n-1)-d flatten before gemm
        assert list(target_shape.input_data) == [1, -1]
        # Connect input of reshape to gemm, then remove reshape
        graph.add_edge(reshape_input, n_gemm, index=0)
        graph.remove_node(node)
    return graph


def get_next_in_chain(
    graph: nx.DiGraph, type_: str, node: Optional[NodeT]
) -> Optional[NodeT]:
    """
    Get a unique user node of the unique output of Node `node`,
    and return it if it has Type `type_`.
    """
    if node is None or len(node.output) != 1:
        return None  # Propagates None; Unique output
    users = list(graph.neighbors(node))
    if len(users) != 1 or users[0].op_type != type_:
        return None  # Unique user of the output; Correct type
    return users[0]


def build_graph_with_mapping(
    graph: nx.DiGraph, node_mapping: Dict[NodeT, EmitNodeT]
) -> nx.DiGraph:
    graph = graph.copy()
    single_node, multi_node = {}, {}
    for replace_node, by_node in node_mapping.items():
        if isinstance(by_node, g.DFGNode):
            single_node[replace_node] = by_node
        else:
            multi_node[replace_node] = by_node
    # We do one-to-many replacements first
    # because their predecessors are specified as onnx nodes.
    for replace_node, subgraph in multi_node.items():
        # Add subgraph itself
        graph = nx.compose(graph, subgraph.subgraph)
        # Add in edges
        graph.add_edges_from(subgraph.entry_edges)
        # Add out edges
        succ = graph.out_edges(replace_node, "index")
        for _, to, index in succ:
            graph.add_edge(subgraph.exit, to, index=index)
        # Remove old node
        graph.remove_node(replace_node)
    # Then do all one-to-one replacements.
    graph = nx.relabel_nodes(graph, single_node)
    return graph


def extract_tensors_from_graph(onnx_graph: GraphT) -> Dict[str, g.TensorNode]:
    tensors = {}
    # parse weight
    weight_cnt = 0
    for weight_tensor in onnx_graph.initializer:
        tensors[weight_tensor.name] = g.WeightTensor(
            weight_tensor, f"weight_{weight_cnt}"
        )
        weight_cnt += 1
    # parse input
    input_cnt = 0
    for input_ in onnx_graph.input:
        if input_.name in tensors:
            continue
        tensors[input_.name] = g.InputTensor(
            input_, get_node_shape(input_), f"input_{input_cnt}"
        )
        input_cnt += 1
    return tensors


def sorted_inputs(graph: nx.DiGraph, node):
    sorted_edges = sorted(graph.in_edges(node, "index"), key=lambda p: p[2])
    return [e[0] for e in sorted_edges]


def draw_graph(graph: nx.DiGraph, output_to):
    from networkx.drawing.nx_agraph import to_agraph

    agraph = to_agraph(graph)
    agraph.layout("dot")
    agraph.draw(output_to)
