from .node import OperatorNode


class ConstantFloat(OperatorNode):
    _arity = 0
    _def_output = "1.0"
    _def_numpy_output = "np.ones(x.shape[0]) * 1.0"
    _def_torch_output = "torch.ones(1).expand(x.shape[0]) * 1.0"


class Add(OperatorNode):
    _arity = 2
    _def_output = "x_0 + x_1"


class Sub(OperatorNode):
    _arity = 2
    _def_output = "x_0 - x_1"


class Mul(OperatorNode):
    _arity = 2
    _def_output = "x_0 * x_1"


class Div(OperatorNode):
    _arity = 2
    _def_output = "x_0 / x_1"


class Pow(OperatorNode):
    _arity = 2
    _def_output = "x_0 ** x_1"
    _def_numpy_output = "np.power(x_0, x_1)"


class Parameter(OperatorNode):
    _arity = 0
    _initial_values = {"<p>": lambda: 1.0}
    _def_output = "<p>"
    _def_numpy_output = "np.ones(x.shape[0]) * <p>"
    _def_torch_output = "torch.ones(1).expand(x.shape[0]) * <p>"
