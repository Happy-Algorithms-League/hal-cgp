import collections
import numpy as np  # noqa: F401
import re

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

from typing import Callable, DefaultDict, Dict, List, Optional, Set

from .node import Node, InputNode, OutputNode, Parameter
from .genome import Genome


class CartesianGraph:
    """Class representing a particular Cartesian graph defined by a
    Genome.
    """

    def __init__(self, genome: Genome) -> None:
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
        self._parameter_names_to_values: DefaultDict

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

    def parse_genome(self, genome: Genome) -> None:
        if genome.dna is None:
            raise RuntimeError("dna not initialized")

        self._n_inputs = genome._n_inputs
        self._n_outputs = genome._n_outputs
        self._n_columns = genome._n_columns
        self._n_rows = genome._n_rows

        # WARNING: creating a reference, not a copy here is essential;
        # accessing missing elements in this DefaultDict during graph
        # construction needs to construct keys with default values
        self._parameter_names_to_values = genome.parameter_names_to_values

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

            for i in node.inputs:
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

    def __call__(self, x: List[float]) -> List[float]:
        # store values of x in input nodes
        for i, xi in enumerate(x):
            assert isinstance(self._nodes[i], InputNode)
            self._nodes[i]._output = xi

        # evaluate active nodes in order
        active_nodes_by_hidden_column_idx = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes_by_hidden_column_idx):
            for node in active_nodes_by_hidden_column_idx[hidden_column_idx]:
                node(x, self)

        return [node._output for node in self.output_nodes]

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
        g = re.findall("<p[0-9]+>", func_str)
        if len(g) != 0:
            for parameter_name in g:
                func_str = func_str.replace(
                    parameter_name, str(self._parameter_names_to_values[parameter_name])
                )
        return func_str

    def to_func(self) -> Callable[[List[float]], List[float]]:
        """Compile the function(s) represented by the graph.

        Generates a definition of the function in Python code and
        executes the function definition to create a Callable.

        Returns
        -------
        Callable
            Callable executing the function(s) represented by the graph.
        """
        self._format_output_str_of_all_nodes()
        s = ", ".join(node.output_str for node in self.output_nodes)
        func_str = f"""\
def _f(x):
    if len(x) != {self._n_inputs}:
        raise ValueError(f'input has length {{len(x)}}, expected {self._n_inputs}')
    return [{s}]
"""
        func_str = self._fill_parameter_values(func_str)
        exec(func_str)
        return locals()["_f"]

    def _format_output_str_numpy_of_all_nodes(self):

        for i, node in enumerate(self.input_nodes):
            node.format_output_str_numpy(self)

        active_nodes = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes):
            for node in active_nodes[hidden_column_idx]:
                node.format_output_str_numpy(self)

    def to_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        """Compile the function(s) represented by the graph to NumPy
        expression(s).

        Generates a definition of the function in Python code and
        executes the function definition to create a Callable
        accepting NumPy arrays.

        Returns
        -------
        Callable
            Callable executing the function(s) represented by the graph.
        """

        self._format_output_str_numpy_of_all_nodes()
        s = ", ".join(node.output_str for node in self.output_nodes)
        func_str = f"""\
def _f(x):
    if len(x.shape) != 2:
        raise ValueError(
            f"input has shape {{tuple(x.shape)}}, expected (<batch_size>, {self._n_inputs})"
        )
    if x.shape[1] != {self._n_inputs}:
        raise ValueError(
            f"input has shape {{tuple(x.shape)}}, expected ({{x.shape[0]}}, {self._n_inputs})"
        )

    return np.stack([{s}], axis=1)
"""
        func_str = self._fill_parameter_values(func_str)
        exec(func_str)

        return locals()["_f"]

    def to_torch(self) -> "torch.nn.Module":
        """Compile the function(s) represented by the graph to a Torch class.

        Generates a definition of the Torch class in Python code and
        executes it to create an instance of the class.

        Returns
        -------
        torch.nn.Module
            Instance of the PyTorch class.
        """
        if not torch_available:
            raise ModuleNotFoundError("No module named 'torch' (extra requirement)")

        for i, node in enumerate(self.input_nodes):
            node.format_output_str_torch(self)

        active_nodes_by_hidden_column_idx = self._determine_active_nodes()
        all_parameter_str = []
        for hidden_column_idx in sorted(active_nodes_by_hidden_column_idx):
            for node in active_nodes_by_hidden_column_idx[hidden_column_idx]:
                node.format_output_str_torch(self)
                if isinstance(node, Parameter):
                    node.format_parameter_str()
                    all_parameter_str.append(node.parameter_str)
        forward_str = ", ".join(node.output_str for node in self.output_nodes)
        class_str = """\
class _C(torch.nn.Module):

    def __init__(self):
        super().__init__()

"""
        for s in all_parameter_str:
            class_str += "        " + s

        func_str = f"""\

    def forward(self, x):
        if len(x.shape) != 2:
            raise ValueError(
                f"input has shape {{tuple(x.shape)}}, expected (<batch_size>, {self._n_inputs})"
            )
        if x.shape[1] != {self._n_inputs}:
            raise ValueError(
                f"input has shape {{tuple(x.shape)}}, expected ({{x.shape[0]}}, {self._n_inputs})"
            )
        return torch.stack([{forward_str}], dim=1)
        """
        class_str += func_str
        class_str = self._fill_parameter_values(class_str)

        exec(class_str)
        exec("_c = _C()")

        return locals()["_c"]

    def to_sympy(self, simplify: Optional[bool] = True,) -> List["sympy_expr.Expr"]:
        """Compile the function(s) represented by the graph to a SymPy expression.

        Generates one SymPy expression for each output node.

        Parameters
        ----------
        simplify : boolean, optional
            Whether to simplify the expression using SymPy's
            simplify() method. Defaults to True.

        Returns
        ----------
        List[sympy.core.expr.Expr]
            List of SymPy expressions.
        """
        if not sympy_available:
            raise ModuleNotFoundError("No module named 'sympy' (extra requirement)")

        self._format_output_str_of_all_nodes()

        sympy_exprs = []
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
            with sympy.evaluate(False):
                sympy_exprs.append(sympy.sympify(s))

        if not simplify:
            return sympy_exprs
        else:  # simplify expression if desired
            for i, expr in enumerate(sympy_exprs):
                sympy_exprs[i] = expr.simplify()
            return sympy_exprs
