import math  # noqa: F401
import re
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Type

from . import node_validation

if TYPE_CHECKING:
    from .cartesian_graph import CartesianGraph

primitives_dict = {}  # maps string of class names to classes


def register(cls: Type["Node"]) -> None:
    """Register a primitive in the global dictionary of primitives

    Parameters
    ----------
    cls : Type[Node]
       Primitive to be registered.

    Returns
    ----------
    None
    """
    name = cls.__name__
    if name not in primitives_dict:
        primitives_dict[name] = cls


class Node:
    """Base class for input/output and hidden nodes.
    """

    _arity: int
    _active: bool = False
    _addresses: List[int]
    _output: float
    _output_str: str
    _idx: int

    def __init__(self, idx: int, addresses: List[int]) -> None:
        """Init function.

        Parameters
        ----------
        idx : int
            id of the node.
        addresses : List[int]
            List of integers specifying the address of input nodes to this node.
        """
        self._idx = idx
        self._addresses = addresses[: self._arity]

        assert idx not in addresses

    def __init_subclass__(cls: Type["Node"]) -> None:
        super().__init_subclass__()
        register(cls)

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def max_arity(self) -> int:
        return len(self._addresses)

    @property
    def addresses(self) -> List[int]:
        return self._addresses

    @property
    def idx(self) -> int:
        return self._idx

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(idx: {self.idx}, active: {self._active}, "
            f"arity: {self._arity}, addresses of inputs {self._addresses}"
        )

    def pretty_str(self, n: int) -> str:
        used_characters = 0
        used_characters += 3  # for two digit idx and following whitespace
        used_characters += 3  # for "active" marker
        # for brackets around addresses, two digits addresses separated by
        # comma and last address without comma
        used_characters += 2 + self.max_arity * 3 - 1

        assert n > used_characters
        name = self.__class__.__name__
        name = name[: n - used_characters]  # cut to correct size

        s = f"{self._idx:02d}"

        if self._active:
            s += " * "
            s += name + " "
            if self._arity > 0:
                s += "("
                for i in range(self._arity):
                    s += f"{self._addresses[i]:02d},"
                s = s[:-1]
                s += ")"
                for i in range(self.max_arity - self._arity):
                    s += "   "
            else:
                s += "  "
                for i in range(self.max_arity):
                    s += "   "
                s = s[:-1]
        else:
            s += "   "
            s += name + " "
            s += "  "
            for i in range(self.max_arity):
                s += "   "
            s = s[:-1]

        return s.ljust(n)

    @property
    def output(self) -> float:
        return self._output

    @property
    def output_str(self) -> str:
        return self._output_str

    def activate(self) -> None:
        """Set node to active.
        """
        self._active = True

    def format_output_str(self, graph: "CartesianGraph") -> None:
        """Format output string of the node.
        """
        raise NotImplementedError()

    def format_output_str_numpy(self, graph):
        """Format output string for NumPy representation.
        """
        raise NotImplementedError()

    def format_output_str_torch(self, graph):
        """Format output string for PyTorch representation.
        """
        raise NotImplementedError()

    def format_output_str_sympy(self, graph):
        """Format output string for SymPy representation.
        """
        raise NotImplementedError()


class OperatorNode(Node):
    """Base class of hidden nodes.

    Subclasses provide the atomic operations of the computational graph.
    """

    _address_names: Tuple[str, ...]
    _parameter_names: Tuple[str, ...]
    _initial_values: Dict[str, Callable[[], float]]

    _def_output: str
    _def_numpy_output: str
    _def_torch_output: str
    _def_sympy_output: str

    _parameter_str: List[str]

    def __init_subclass__(cls: Type["OperatorNode"]) -> None:
        super().__init_subclass__()
        OperatorNode._extract_input_names_from_def_output(cls)
        OperatorNode._extract_parameter_names_from_def_output(cls)

        node_validation.check_to_func(cls)
        node_validation.check_to_numpy(cls)
        node_validation.check_to_torch(cls)
        node_validation.check_to_sympy(cls)

    @classmethod
    def _extract_input_names_from_def_output(self, cls: Type["OperatorNode"]) -> None:
        g = set(re.findall("x_[0-9]+", cls._def_output))

        if not len(g) == cls._arity:
            raise RuntimeError(f'wrong number of inputs defined in OperatorNode "{cls.__name__}"')

        cls._address_names = tuple(g)

    @classmethod
    def _extract_parameter_names_from_def_output(self, cls: Type["OperatorNode"]) -> None:
        g = re.findall("<[a-z]+>", cls._def_output)
        cls._parameter_names = tuple(g)

    @staticmethod
    def _extract_index_from_address_name(address_name: str) -> int:
        return int(address_name.split("_")[1])

    @classmethod
    def initial_value(cls, parameter_name: str) -> float:
        parameter_prefix: str = re.findall("([a-z]+)[0-9]+", parameter_name)[0]
        return cls._initial_values["<" + parameter_prefix + ">"]()

    @property
    def parameter_str(self) -> List[str]:
        return self._parameter_str

    def _replace_address_names(self, output_str: str, graph: "CartesianGraph") -> str:
        for address_name in self._address_names:
            idx = self._extract_index_from_address_name(address_name)
            output_str = output_str.replace(address_name, graph[self._addresses[idx]].output_str)
        return output_str

    def _replace_parameter_names(self, output_str: str, graph: "CartesianGraph") -> str:
        for parameter_name in self._parameter_names:
            parameter_name_with_idx = parameter_name[1:-1] + str(self._idx)
            output_str = output_str.replace(parameter_name, "<" + parameter_name_with_idx + ">")
        return output_str

    def _format_output_str(self, output_str: str, graph: "CartesianGraph") -> str:
        output_str = str(output_str)
        output_str = self._replace_address_names(output_str, graph)
        output_str = self._replace_parameter_names(output_str, graph)
        return "(" + output_str + ")"

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = self._format_output_str(self._def_output, graph)

    def format_output_str_numpy(self, graph: "CartesianGraph") -> None:
        if not hasattr(self, "_def_numpy_output"):
            self.format_output_str(graph)
        else:
            self._output_str = self._format_output_str(self._def_numpy_output, graph)

    def format_output_str_torch(self, graph: "CartesianGraph") -> None:
        if not hasattr(self, "_def_torch_output"):
            output_str = self._format_output_str(self._def_output, graph)
        else:
            output_str = self._format_output_str(self._def_torch_output, graph)
        self._output_str = self._replace_parameter_names_in_output_str_with_members(output_str)

    def _replace_parameter_names_in_output_str_with_members(self, output_str: str) -> str:
        g = re.findall("<([a-z]+[0-9]+)>", output_str)
        for parameter_name_with_idx in g:
            output_str = output_str.replace(
                "<" + parameter_name_with_idx + ">", "self._" + parameter_name_with_idx
            )
        return output_str

    def format_output_str_sympy(self, graph: "CartesianGraph") -> None:
        if not hasattr(self, "_def_sympy_output"):
            self.format_output_str(graph)
        else:
            self._output_str = self._format_output_str(self._def_sympy_output, graph)

    def format_parameter_str(self) -> None:
        parameter_str = []
        for parameter_name in self._parameter_names:
            parameter_name_with_idx = parameter_name[1:-1] + str(self._idx)
            parameter_str.append(
                f"self._{parameter_name_with_idx} = torch.nn.Parameter("
                + f"torch.DoubleTensor([<{parameter_name_with_idx}>]))"
            )
        self._parameter_str = parameter_str
