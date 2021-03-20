from typing import TYPE_CHECKING, List

from .node import Node

if TYPE_CHECKING:
    from .cartesian_graph import CartesianGraph


class InputNode(Node):
    """An input node of the computational graph.
    """

    _arity = 0

    def __init__(self, idx: int, input_nodes: List[int]) -> None:
        super().__init__(idx, input_nodes)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        assert False

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = f"x[{self._idx}]"

    def format_output_str_numpy(self, graph: "CartesianGraph") -> None:
        self.format_output_str(graph)

    def format_output_str_torch(self, graph: "CartesianGraph") -> None:
        self._output_str = f"x[:, {self._idx}]"

    def format_output_str_sympy(self, graph: "CartesianGraph") -> None:
        self.format_output_str(graph)


class OutputNode(Node):
    """An output node of the computational graph.
    """

    _arity = 1

    def __init__(self, idx: int, input_nodes: List[int]) -> None:
        super().__init__(idx, input_nodes)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        self._output = graph[self._addresses[0]].output

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = f"{graph[self._addresses[0]].output_str}"

    def format_output_str_numpy(self, graph: "CartesianGraph") -> None:
        self.format_output_str(graph)

    def format_output_str_torch(self, graph: "CartesianGraph") -> None:
        self.format_output_str(graph)

    def format_output_str_sympy(self, graph: "CartesianGraph") -> None:
        self.format_output_str(graph)
