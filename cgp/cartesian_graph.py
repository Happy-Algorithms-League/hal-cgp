import collections
import copy
import math  # noqa: F401
import re
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Union

import numpy as np  # noqa: F401

from .node import Node, OperatorNode
from .node_input_output import InputNode, OutputNode

try:
    import sympy
    from sympy.core import expr as sympy_expr  # noqa: F401

    sympy_available = True
except ModuleNotFoundError:
    sympy_available = False

try:
    import torch  # noqa: F401

    torch_available = True
except ModuleNotFoundError:
    torch_available = False


if TYPE_CHECKING:
    from .genome import Genome


CUSTOM_ATOMIC_OPERATORS = {}


def atomic_operator(f: Callable) -> Callable:
    CUSTOM_ATOMIC_OPERATORS[f.__name__] = f
    return f


class CartesianGraph:
    """Class representing a particular Cartesian graph defined by a
    Genome.
    """

    def __init__(self, genome: "Genome") -> None:
        """Init function.

        Parameters
        ----------
        genome: Genome
            Genome defining graph connectivity and node operations.
        """
        self._n_inputs: int
        self._n_outputs: int
        self._n_columns: int
        self._n_rows: int
        self._nodes: List
        self._parameter_names_to_values: Dict[str, float]

        self.parse_genome(genome)

    def __repr__(self) -> str:
        return "CartesianGraph(" + str(self._nodes) + ")"

    def print_active_nodes(self) -> str:
        """Print a representation of all active nodes in the graph.
        """
        return "CartesianGraph(" + str([node for node in self._nodes if node._active]) + ")"

    def pretty_str(self) -> str:
        """Print a pretty representation of the Cartesian graph.
        """
        n_characters = 24

        def pretty_node_str(node: Node) -> str:
            s = node.pretty_str(n_characters)
            assert len(s) == n_characters
            return s

        def empty_node_str() -> str:
            return " " * n_characters

        s = "\n"

        for row in range(max(self._n_inputs, self._n_rows)):
            for column in range(-1, self._n_columns + 1):

                if column == -1:
                    if row < self._n_inputs:
                        s += pretty_node_str(self.input_nodes[row])
                    else:
                        s += empty_node_str()
                    s += "\t"

                elif column < self._n_columns:
                    if row < self._n_rows:
                        s += pretty_node_str(self.hidden_nodes[row + column * self._n_rows])
                    else:
                        s += empty_node_str()
                    s += "\t"
                else:
                    if row < self._n_outputs:
                        s += pretty_node_str(self.output_nodes[row])
                    else:
                        s += empty_node_str()
                    s += "\t"

            s += "\n"

        return s

    def parse_genome(self, genome: "Genome") -> None:
        if genome.dna is None:
            raise RuntimeError("dna not initialized")

        self._n_inputs = genome._n_inputs
        self._n_outputs = genome._n_outputs
        self._n_columns = genome._n_columns
        self._n_rows = genome._n_rows
        self._parameter_names_to_values = copy.deepcopy(genome._parameter_names_to_values)

        self._nodes = []

        idx = 0
        for region_idx, input_region in genome.iter_input_regions():
            self._nodes.append(InputNode(idx, input_region[1:]))
            idx += 1

        for region_idx, hidden_region in genome.iter_hidden_regions():
            self._nodes.append(genome.primitives[hidden_region[0]](idx, hidden_region[1:]))
            idx += 1

        for region_idx, output_region in genome.iter_output_regions():
            self._nodes.append(OutputNode(idx, output_region[1:]))
            idx += 1

        self._determine_active_nodes()

    def _hidden_column_idx(self, idx: int) -> int:
        return (idx - self._n_inputs) // self._n_rows

    @property
    def input_nodes(self) -> List[Node]:
        return self._nodes[: self._n_inputs]

    @property
    def hidden_nodes(self) -> List[Node]:
        return self._nodes[self._n_inputs : -self._n_outputs]

    @property
    def output_nodes(self) -> List[Node]:
        return self._nodes[-self._n_outputs :]

    def _determine_active_nodes(self) -> Dict[int, Set[Node]]:
        """Determine the active nodes in the graph.

        Starting from the output nodes, we work backward through the
        graph to determine all hidden nodes which are encountered on
        the path from input to output nodes. For each hidden column
        index we thus construct a set of active nodes. Since nodes can
        only receive input from previous layers, a forward pass can
        easily work through the columns in order, updating only the
        active nodes.

        Returns
        -------
        Dict[int, Set[Node]]
            Dictionary mapping colum indices to sets of active nodes.

        """

        # we use sets to make sure each node index is only stored once
        active_nodes_by_hidden_column_idx = collections.defaultdict(set)
        nodes_to_process = list(self.output_nodes)

        while len(nodes_to_process) > 0:  # process active nodes in a stack-based fashion

            node = nodes_to_process.pop()
            node.activate()
            if node in self.input_nodes:
                continue
            active_nodes_by_hidden_column_idx[self._hidden_column_idx(node.idx)].add(node)

            for i in node.addresses:
                nodes_to_process.append(self._nodes[i])

        return active_nodes_by_hidden_column_idx

    def determine_active_regions(self) -> List[int]:
        """Determine the active regions in the computational graph.

        Returns
        -------
        List[int]
            List of ids of the active nodes.
        """
        active_regions = []
        active_nodes_by_hidden_column_idx = self._determine_active_nodes()
        for column_idx in active_nodes_by_hidden_column_idx:
            for node in active_nodes_by_hidden_column_idx[column_idx]:
                active_regions.append(node.idx)

        return active_regions

    def __getitem__(self, key: int) -> Node:
        return self._nodes[key]

    def to_str(self) -> str:

        self._format_output_str_of_all_nodes()
        out_str = ", ".join(node.output_str for node in self.output_nodes)
        return f"[{out_str}]"

    def _format_output_str_of_all_nodes(self) -> None:

        for i, node in enumerate(self.input_nodes):
            node.format_output_str(self)

        active_nodes = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes):
            for node in active_nodes[hidden_column_idx]:
                node.format_output_str(self)

    def _fill_parameter_values(self, func_str: str) -> str:
        g = re.findall("<[a-z]+[0-9]+>", func_str)
        if len(g) != 0:
            for parameter_name in g:
                func_str = func_str.replace(
                    parameter_name, str(self._parameter_names_to_values[parameter_name])
                )
        return func_str

    def to_func(self) -> Callable[..., List[float]]:
        """Create a Python callable implementing the function described by
        this graph.

        The returned callable expects as many arguments as the number
        of inputs defined in the genome. The function returns a tuple
        with length equal to the number of outputs defined in the
        genome. For convenience, if only a single output is defined
        the function will *not* return a tuple but only its first
        element.

        Returns
        -------
        Callable

        """
        self._format_output_str_of_all_nodes()
        s = ", ".join(node.output_str for node in self.output_nodes)
        func_str = f"""\
def _f(*x):
    if len(x) != {self._n_inputs}:
        raise ValueError(f'input has length {{len(x)}}, expected {self._n_inputs}')

    res = [{s}]

    if len(res) == 1:
        return res[0]
    else:
        return res
"""
        func_str = self._fill_parameter_values(func_str)
        exec(func_str, {**globals(), **CUSTOM_ATOMIC_OPERATORS}, locals())
        return locals()["_f"]

    def _format_output_str_numpy_of_all_nodes(self):

        for i, node in enumerate(self.input_nodes):
            node.format_output_str_numpy(self)

        active_nodes = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes):
            for node in active_nodes[hidden_column_idx]:
                node.format_output_str_numpy(self)

    def to_numpy(self) -> Callable[..., List[np.ndarray]]:
        """Create a NumPy-array-compatible Python callable implementing the
        function described by this graph.

        The returned callable expects as many arguments as the number
        of inputs defined in the genome. Every argument needs to be a
        NumPy array of equal length. The function returns a tuple with
        length equal to the number of outputs defined in the
        genome. Each element will have the same length as the input
        arrays. For convenience, if only a single output is defined
        the function will *not* return a tuple but only its first
        element.

        Returns
        -------
        Callable

        """

        self._format_output_str_numpy_of_all_nodes()
        s = ", ".join(node.output_str for node in self.output_nodes)
        func_str = f"""\
def _f(*x):
    if len(x) != {self._n_inputs}:
        raise ValueError(f'input has length {{len(x)}}, expected {self._n_inputs}')

    res = [{s}]

    if len(res) == 1:
        return res[0]
    else:
        return res
"""
        func_str = self._fill_parameter_values(func_str)
        exec(func_str, {**globals(), **CUSTOM_ATOMIC_OPERATORS}, locals())

        return locals()["_f"]

    def to_torch(self) -> "torch.nn.Module":
        """Create a Torch nn.Module instance implementing the function defined
        by this graph.

        The generated instance will have a `forward` method accepting
        Torch tensor of dimension (<batch size>, n_inputs) and
        returning a tensor of dimension (<batch_size>, n_outputs).

        Returns
        -------
        torch.nn.Module

        """
        if not torch_available:
            raise ModuleNotFoundError("No module named 'torch' (extra requirement)")

        for i, node in enumerate(self.input_nodes):
            node.format_output_str_torch(self)

        active_nodes_by_hidden_column_idx = self._determine_active_nodes()
        all_parameter_str: List[List[str]] = []
        for hidden_column_idx in sorted(active_nodes_by_hidden_column_idx):
            for node in active_nodes_by_hidden_column_idx[hidden_column_idx]:
                node.format_output_str_torch(self)
                if isinstance(node, OperatorNode):
                    if len(node._parameter_names) > 0:
                        node.format_parameter_str()
                        all_parameter_str.append(node.parameter_str)

        forward_str = ", ".join(node.output_str for node in self.output_nodes)
        class_str = """\
class _C(torch.nn.Module):

    def __init__(self):
        super().__init__()

"""
        for parameter_str in all_parameter_str:
            for s in parameter_str:
                class_str += "        " + s + "\n"

        func_str = f"""\

    def forward(self, x):
        if (len(x.shape) != 2) or (x.shape[1] != {self._n_inputs}):
            raise ValueError(
                f"input has shape {{tuple(x.shape)}}, expected (<batch_size>, {self._n_inputs})"
            )
        return torch.stack([{forward_str}], dim=1)
        """
        class_str += func_str
        class_str = self._fill_parameter_values(class_str)

        exec(class_str, {**globals(), **CUSTOM_ATOMIC_OPERATORS}, locals())
        exec("_c = _C()")

        return locals()["_c"]

    def _format_output_str_sympy_of_all_nodes(self):

        for i, node in enumerate(self.input_nodes):
            node.format_output_str_sympy(self)

        active_nodes = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes):
            for node in active_nodes[hidden_column_idx]:
                node.format_output_str_sympy(self)

    def to_sympy(
        self, simplify: Optional[bool] = True
    ) -> Union["sympy_expr.Expr", List["sympy_expr.Expr"]]:
        """Create SymPy expression(s) representing the function(s) described
        by this graph.

        Returns a list of SymPy expressions, one for each output
        node. For convenience, if only one output node is defined, it
        directly returns its expression.

        Parameters
        ----------
        simplify : boolean, optional
            Whether to simplify the expression using SymPy's
            simplify() method. Defaults to True.

        Returns
        ----------
        List[sympy.core.expr.Expr] or sympy.core.expr.Expr
            List of SymPy expressions or single expression.

        """

        if not sympy_available:
            raise ModuleNotFoundError("No module named 'sympy' (extra requirement)")

        self._format_output_str_sympy_of_all_nodes()

        sympy_exprs: List = []
        for output_node in self.output_nodes:

            # replace all input-variable strings with sympy-compatible symbol
            # strings (i.e., x[0] -> x_0)
            s = output_node.output_str
            for input_node in self.input_nodes:
                s = s.replace(
                    input_node.output_str, input_node.output_str.replace("[", "_").replace("]", "")
                )

            s = self._fill_parameter_values(s)
            # to get an expression that reflects the computational graph,
            # sympy should not automatically simplify the expression
            sympy_exprs.append(sympy.sympify(s, evaluate=False))

        if simplify:
            for i, expr in enumerate(sympy_exprs):
                try:
                    sympy_exprs[i] = expr.simplify()
                except TypeError:
                    RuntimeWarning(f"SymPy could not simplify expression: {expr}")

        # if the genome encodes only a single function we directly
        # return the sympy expression instead of a list of length 1
        if len(sympy_exprs) == 1:
            return sympy_exprs[0]
        else:
            return sympy_exprs
