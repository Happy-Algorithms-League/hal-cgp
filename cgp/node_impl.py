from .node import OperatorNode


class ConstantFloat(OperatorNode):
    """A node with a constant output."""

    _arity = 0
    _def_output = "1.0"
    _def_numpy_output = "np.ones(len(x[0])) * 1.0"
    _def_torch_output = "torch.ones(1).expand(x.shape[0]) * 1.0"


class Add(OperatorNode):
    """A node that adds its two inputs."""

    _arity = 2
    _def_output = "x_0 + x_1"


class Sub(OperatorNode):
    """A node that substracts its second from its first input."""

    _arity = 2
    _def_output = "x_0 - x_1"


class Mul(OperatorNode):
    """A node that multiplies its two inputs."""

    _arity = 2
    _def_output = "x_0 * x_1"


class Div(OperatorNode):
    """A node that devides its first by its second input."""

    _arity = 2
    _def_output = "x_0 / x_1"


class Pow(OperatorNode):
    """A node that raises its first to the power of its second input."""

    _arity = 2
    _def_output = "x_0 ** x_1"
    _def_numpy_output = "np.power(x_0, x_1)"


class Parameter(OperatorNode):
    """A node that provides a parametrized constant output.

    The value of the parameter can be adapted via local search and is
    passed on from parents to their offspring.

    """

    _arity = 0
    _initial_values = {"<p>": lambda: 1.0}
    _def_output = "<p>"
    _def_numpy_output = "np.ones(len(x[0])) * <p>"
    _def_torch_output = "torch.ones(1).expand(x.shape[0]) * <p>"


class IfElse(OperatorNode):
    """A node that outputs the value of its second input if its first input
    is non-negative, and the value of its third input otherwise."""

    _arity = 3
    _def_output = "x_1 if x_0 >= 0 else x_2"
    _def_numpy_output = "np.piecewise(x_0, [x_0 >= 0, x_0 < 0], [x_1[x_0 >= 0] , x_2[x_0 < 0]])"
    _def_sympy_output = "Piecewise((x_1, x_0 >= 0), (x_2, x_0 < 0))"
    _def_torch_output = "torch.where(x_0 >= 0, x_1, x_2)"
