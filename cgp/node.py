from typing import List, Type, TYPE_CHECKING

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
    """Base class for primitive functions used in Cartesian computational graphs.
    """

    _arity: int
    _active: bool = False
    _inputs: List[int]
    _output: float
    _output_str: str
    _parameter_str: str
    _idx: int

    def __init__(self, idx: int, inputs: List[int]) -> None:
        """Init function.

        Parameters
        ----------
        idx : int
            id of the node.
        inputs : List[int]
            List of integers specifying the input nodes to this node.
        """
        self._idx = idx
        self._inputs = inputs[: self._arity]

        assert idx not in inputs

    def __init_subclass__(cls: Type["Node"]) -> None:
        super().__init_subclass__()
        register(cls)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        raise NotImplementedError

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def max_arity(self) -> int:
        return len(self._inputs)

    @property
    def inputs(self) -> List[int]:
        return self._inputs

    @property
    def idx(self) -> int:
        return self._idx

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(idx: {self.idx}, active: {self._active}, "
            f"arity: {self._arity}, inputs {self._inputs}, output {self._output})"
        )

    def pretty_str(self, n: int) -> str:
        used_characters = 0
        used_characters += 3  # for two digit idx and following whitespace
        used_characters += 3  # for "active" marker
        # for brackets around inputs, two digits inputs separated by
        # comma and last input without comma
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
                    s += f"{self._inputs[i]:02d},"
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

        If format_output_str_numpy implementation is not provided, use
        standard output_str.
        """
        self.format_output_str(graph)

    def format_output_str_torch(self, graph):
        """Format output string for torch representation.

        If format_output_str_torch implementation is not provided, use
        standard output_str.
        """
        self.format_output_str(graph)

    def format_parameter_str(self) -> None:
        raise NotImplementedError

    @property
    def output_str(self) -> str:
        return self._output_str

    @property
    def parameter_str(self) -> str:
        return self._parameter_str


class Add(Node):
    """Node representing addition.
    """

    _arity = 2

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        self._output = graph[self._inputs[0]].output + graph[self._inputs[1]].output

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} + {graph[self._inputs[1]].output_str})"
        )


class Sub(Node):
    """Node representing subtraction.
    """

    _arity = 2

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        self._output = graph[self._inputs[0]].output - graph[self._inputs[1]].output

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} - {graph[self._inputs[1]].output_str})"
        )


class Mul(Node):
    """Node representing multiplication.
    """

    _arity = 2

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        self._output = graph[self._inputs[0]].output * graph[self._inputs[1]].output

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} * {graph[self._inputs[1]].output_str})"
        )


class Div(Node):
    """Node representing division.
    """

    _arity = 2

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:

        self._output = graph[self._inputs[0]].output / graph[self._inputs[1]].output

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} / {graph[self._inputs[1]].output_str})"
        )


class ConstantFloat(Node):
    """Node representing a constant float number.
    """

    _arity = 0

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

        self._output = 1.0

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        pass

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = f"{self._output}"

    def format_output_str_numpy(self, graph: "CartesianGraph") -> None:
        self._output_str = f"np.ones(x.shape[0]) * {self._output}"

    def format_output_str_torch(self, graph: "CartesianGraph") -> None:
        self._output_str = f"torch.ones(1).expand(x.shape[0]) * {self._output}"


class Parameter(Node):
    """Node representing a scalar variable. Its value is stored in the
    individual holding the corresponding genome.
    """

    _arity = 0

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        pass

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = f"<p{self._idx}>"

    def format_output_str_numpy(self, graph: "CartesianGraph") -> None:
        self._output_str = f"np.ones(x.shape[0]) * <p{self._idx}>"

    def format_output_str_torch(self, graph: "CartesianGraph") -> None:
        self._output_str = f"self._p{self._idx}.expand(x.shape[0])"

    def format_parameter_str(self) -> None:
        self._parameter_str = (
            f"self._p{self._idx} = torch.nn.Parameter(torch.DoubleTensor([<p{self._idx}>]))\n"
        )


class InputNode(Node):
    """Node representing a generic input node.
    """

    _arity = 0

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        assert False

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = f"x[{self._idx}]"

    def format_output_str_numpy(self, graph: "CartesianGraph") -> None:
        self._output_str = f"x[:, {self._idx}]"

    def format_output_str_torch(self, graph: "CartesianGraph") -> None:
        self._output_str = f"x[:, {self._idx}]"


class OutputNode(Node):
    """Node representing a generic output node.
    """

    _arity = 1

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        self._output = graph[self._inputs[0]].output

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = f"{graph[self._inputs[0]].output_str}"


class Pow(Node):
    """Node representing the power operation.
    """

    _arity = 2

    def __init__(self, idx: int, inputs: List[int]) -> None:
        super().__init__(idx, inputs)

    def __call__(self, x: List[float], graph: "CartesianGraph") -> None:
        self._output = graph[self._inputs[0]].output ** graph[self._inputs[1]].output

    def format_output_str(self, graph: "CartesianGraph") -> None:
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} ** {graph[self._inputs[1]].output_str})"
        )
