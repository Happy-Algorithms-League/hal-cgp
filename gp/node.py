primitives_dict = {}  # maps string of class names to classes


def register(cls):
    """Register a primitive in the global dictionary of primitives

    Parameters
    ----------
    cls : gp.CPGNode
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

    _arity = None
    _active = False
    _inputs = None
    _output = None
    _idx = None
    _is_parameter = False

    def __init__(self, idx, inputs):
        """Init function.

        Parameters
        ----------
        idx : int
            id of the node.
        inputs : List[int]
            List of integers specifying the input nodes to this node.
        """
        self._idx = idx
        self._inputs = inputs

        assert idx not in inputs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        register(cls)

    @property
    def arity(self):
        return self._arity

    @property
    def max_arity(self):
        return len(self._inputs)

    @property
    def inputs(self):
        return self._inputs[: self._arity]

    @property
    def idx(self):
        return self._idx

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(idx: {self.idx}, active: {self._active}, "
            f"arity: {self._arity}, inputs {self._inputs}, output {self._output})"
        )

    def pretty_str(self, n):
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
    def output(self):
        return self._output

    def activate(self):
        """Set node to active.
        """
        self._active = True

    def format_output_str(self, graph):
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

    def format_parameter_str(self):
        raise NotImplementedError

    @property
    def output_str(self):
        return self._output_str

    @property
    def is_parameter(self):
        return self._is_parameter

    @property
    def parameter_str(self):
        return self._parameter_str


class Add(Node):
    """Node representing addition.
    """

    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output + graph[self._inputs[1]].output

    def format_output_str(self, graph):
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} + {graph[self._inputs[1]].output_str})"
        )


class Sub(Node):
    """Node representing subtraction.
    """

    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output - graph[self._inputs[1]].output

    def format_output_str(self, graph):
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} - {graph[self._inputs[1]].output_str})"
        )


class Mul(Node):
    """Node representing multiplication.
    """

    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output * graph[self._inputs[1]].output

    def format_output_str(self, graph):
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} * {graph[self._inputs[1]].output_str})"
        )


class Div(Node):
    """Node representing division.
    """

    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

    def __call__(self, x, graph):

        self._output = graph[self._inputs[0]].output / graph[self._inputs[1]].output

    def format_output_str(self, graph):
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} / {graph[self._inputs[1]].output_str})"
        )


class ConstantFloat(Node):
    """Node representing a constant float number.
    """

    _arity = 0
    _is_parameter = True

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._output = 1.0

    def __call__(self, x, graph):
        pass

    def format_output_str(self, graph):
        self._output_str = f"{self._output}"

    def format_output_str_numpy(self, graph):
        self._output_str = f"np.ones(x.shape[0]) * {self._output}"

    def format_output_str_torch(self, graph):
        self._output_str = f"self._p{self._idx}.expand(x.shape[0])"

    def format_parameter_str(self):
        self._parameter_str = (
            f"self._p{self._idx} = torch.nn.Parameter(torch.Tensor([{self._output}]))\n"
        )


class InputNode(Node):
    """Node representing a generic input node.
    """

    _arity = 0

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

    def __call__(self, x, graph):
        assert False

    def format_output_str(self, graph):
        self._output_str = f"x[{self._idx}]"

    def format_output_str_numpy(self, graph):
        self._output_str = f"x[:, {self._idx}]"

    def format_output_str_torch(self, graph):
        self._output_str = f"x[:, {self._idx}]"


class OutputNode(Node):
    """Node representing a generic output node.
    """

    _arity = 1

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output

    def format_output_str(self, graph):
        self._output_str = f"{graph[self._inputs[0]].output_str}"


class Pow(Node):
    """Node representing the power operation.
    """

    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output ** graph[self._inputs[1]].output

    def format_output_str(self, graph):
        self._output_str = (
            f"({graph[self._inputs[0]].output_str} ** {graph[self._inputs[1]].output_str})"
        )
