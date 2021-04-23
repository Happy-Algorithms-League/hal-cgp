import numpy as np
import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_NON_CODING_GENE, ID_OUTPUT_NODE


def test_addresses_are_cut_to_match_arity():
    """Test that even if a list of addresses longer than the node arity is
    provided, Node.addresses only returns the initial <arity> addresses,
    ignoring the inactive genes.

    """
    idx = 0
    addresses = [1, 2, 3, 4]

    node = cgp.ConstantFloat(idx, addresses)
    assert node.addresses == []

    node = cgp.node_input_output.OutputNode(idx, addresses)
    assert node.addresses == addresses[:1]

    node = cgp.Add(idx, addresses)
    assert node.addresses == addresses[:2]


def _test_to_x_compilations(
    genome,
    x,
    y_target,
    *,
    test_to_func=True,
    test_to_numpy=True,
    test_to_torch=True,
    test_to_sympy=True,
):
    if test_to_func:
        _test_to_func(genome, x, y_target)
    if test_to_numpy:
        _test_to_numpy(genome, x, y_target)
    if test_to_torch:
        _test_to_torch(genome, x, y_target)
    if test_to_sympy:
        _test_to_sympy(genome, x, y_target)


def _test_to_func(genome, x, y_target):
    graph = cgp.CartesianGraph(genome)
    assert graph.to_func()(*x) == pytest.approx(y_target)


def _test_to_numpy(genome, x, y_target):
    graph = cgp.CartesianGraph(genome)
    args = [np.array([xi]) for xi in x]
    assert graph.to_numpy()(*args) == pytest.approx(y_target)


def _test_to_torch(genome, x, y_target):
    torch = pytest.importorskip("torch")
    graph = cgp.CartesianGraph(genome)
    assert graph.to_torch()(torch.Tensor(x).reshape(1, -1)) == pytest.approx(y_target)


def _test_to_sympy(genome, x, y_target):
    pytest.importorskip("sympy")
    graph = cgp.CartesianGraph(genome)
    assert graph.to_sympy().subs({f"x_{i}": x[i] for i in range(len(x))}) == pytest.approx(
        y_target
    )


def test_add():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"], params["n_outputs"], params["n_columns"], params["n_rows"], primitives,
    )
    # f(x) = x[0] + x[1]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]

    x = [5.0, 1.5]
    y_target = x[0] + x[1]

    _test_to_x_compilations(genome, x, y_target)


def test_sub():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1}

    primitives = (cgp.Sub,)
    genome = cgp.Genome(
        params["n_inputs"], params["n_outputs"], params["n_columns"], params["n_rows"], primitives,
    )
    # f(x) = x[0] - x[1]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]

    x = [5.0, 1.5]
    y_target = x[0] - x[1]

    _test_to_x_compilations(genome, x, y_target)


def test_mul():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1}

    primitives = (cgp.Mul,)
    genome = cgp.Genome(
        params["n_inputs"], params["n_outputs"], params["n_columns"], params["n_rows"], primitives,
    )
    # f(x) = x[0] * x[1]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]

    x = [5.0, 1.5]
    y_target = x[0] * x[1]

    _test_to_x_compilations(genome, x, y_target)


def test_div():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1}

    primitives = (cgp.Div,)
    genome = cgp.Genome(
        params["n_inputs"], params["n_outputs"], params["n_columns"], params["n_rows"], primitives,
    )
    # f(x) = x[0] / x[1]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]

    x = [5.0, 1.5]
    y_target = x[0] / x[1]

    _test_to_x_compilations(genome, x, y_target)


def test_pow():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1}

    primitives = (cgp.Pow,)
    genome = cgp.Genome(
        params["n_inputs"], params["n_outputs"], params["n_columns"], params["n_rows"], primitives,
    )
    # f(x) = x[0] ** x[1]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]

    x = [5.0, 1.5]
    y_target = x[0] ** x[1]

    _test_to_x_compilations(genome, x, y_target)


def test_constant_float():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1}

    primitives = (cgp.ConstantFloat,)
    # f(x) = c
    genome = cgp.Genome(
        params["n_inputs"], params["n_outputs"], params["n_columns"], params["n_rows"], primitives,
    )
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        0,
        0,
        ID_OUTPUT_NODE,
        2,
    ]

    x = [1.0, 2.0]
    y_target = 1.0  # by default the output value of the ConstantFloat node is 1.0

    _test_to_x_compilations(genome, x, y_target)


def test_parameter():
    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
    }
    primitives = (cgp.Parameter,)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = c
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]

    x = [1.0]
    y_target = 1.0  # by default the output value of the Parameter node is 1.0

    _test_to_x_compilations(genome, x, y_target)


def test_parameter_w_custom_initial_value():
    initial_value = 3.1415

    class CustomParameter(cgp.Parameter):
        _initial_values = {"<p>": lambda: initial_value}

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
    }
    primitives = (CustomParameter,)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = c
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]

    x = [1.0]
    y_target = initial_value

    _test_to_x_compilations(genome, x, y_target)


def test_parameter_two_nodes():
    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 1,
    }
    primitives = (cgp.Parameter, cgp.Add)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = c1 + c2
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        2,
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
    ]

    x = [1.0]
    # by default the output value of the Parameter node is 1.0,
    # hence the sum of two Parameter nodes is 2.0
    y_target = 2.0

    _test_to_x_compilations(genome, x, y_target)


def test_parameter_w_random_initial_value(rng_seed):
    np.random.seed(rng_seed)

    min_val = 0.5
    max_val = 1.5

    class CustomParameter(cgp.Parameter):
        _initial_values = {"<p>": lambda: np.random.uniform(min_val, max_val)}

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
    }
    primitives = (CustomParameter,)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = c
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]
    f = cgp.CartesianGraph(genome).to_func()
    y = f(0.0)

    assert min_val <= y
    assert y <= max_val
    assert y != pytest.approx(1.0)


def test_multiple_parameters_per_node():

    p = 3.1415
    q = 2.7128

    class DoubleParameter(cgp.OperatorNode):
        _arity = 0
        _initial_values = {"<p>": lambda: p, "<q>": lambda: q}
        _def_output = "<p> + <q>"
        _def_numpy_output = "np.ones(len(x[0])) * (<p> + <q>)"
        _def_torch_output = "torch.ones(1).expand(x.shape[0]) * (<p> + <q>)"

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
    }
    primitives = (DoubleParameter,)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = p + q
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]
    f = cgp.CartesianGraph(genome).to_func()
    y = f(0.0)

    assert y == pytest.approx(p + q)


def test_reset_parameters_upon_creation_of_node(rng):
    class CustomParameter(cgp.Parameter):
        _initial_values = {"<p>": lambda: np.pi}

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
    }
    primitives = (CustomParameter, CustomParameter)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = p
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]
    genome._parameter_names_to_values["<p1>"] = 1.0

    f = cgp.CartesianGraph(genome).to_func()
    y = f(0.0)
    assert y == pytest.approx(1.0)

    # now mutate the genome, since there is only one other option for
    # the hidden node, a new CustomParameter node will be created;
    # creating this new node should reset the parameter to its initial
    # value; after mutation we recreate the correct graph connectivity
    # manually
    genome.mutate(1.0, rng)
    # f(x) = p
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]

    f = cgp.CartesianGraph(genome).to_func()
    y = f(0.0)
    assert y == pytest.approx(np.pi)


def test_if_else_operator():

    genome_params = {
        "n_inputs": 3,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
    }
    primitives = (cgp.IfElse,)
    genome = cgp.Genome(**genome_params, primitives=primitives)

    # f(x_0, x_1, x_2) = {x_1 if x_0 >= 0, x_2 otherwise}
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,  # function gene
        0,  # first gene is address of first input node
        1,  # second gene is address of second input node
        2,  # third gene is address of third input node
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
    ]

    x_0 = [1.0, 10.0, -20.0]
    y_target_0 = 10.0
    _test_to_x_compilations(genome, x_0, y_target_0)

    x_1 = [0.0, 10.0, -20.0]
    y_target_1 = 10.0
    _test_to_x_compilations(genome, x_1, y_target_1)

    x_2 = [-1.0, 10.0, -20.0]
    y_target_2 = -20.0
    _test_to_x_compilations(genome, x_2, y_target_2)


def test_raise_broken_def_output():
    with pytest.raises(SyntaxError):

        class CustomAdd(cgp.OperatorNode):
            _arity = 2
            _def_output = "x_0 +/ x_1"


def test_raise_broken_def_numpy_output():
    with pytest.raises(ValueError):

        class CustomAdd(cgp.OperatorNode):
            _arity = 2
            _def_output = "x_0 + x_1"
            _def_numpy_output = "np.add(x_0 + x_1)"


def test_raise_broken_def_torch_output():
    pytest.importorskip("torch")
    with pytest.raises(TypeError):

        class CustomAdd(cgp.OperatorNode):
            _arity = 2
            _def_output = "x_0 + x_1"
            _def_torch_output = "torch.add(x_0 + x_1)"


def test_raise_broken_def_sympy_output():
    sympy = pytest.importorskip("sympy")
    with pytest.raises(sympy.SympifyError):

        class CustomAdd(cgp.OperatorNode):
            _arity = 2
            _def_output = "x_0 + x_1"
            _def_sympy_output = "x_0 +/ x_1"


def test_repr():
    idx = 0
    addresses = [1, 2, 3, 4]

    # Test example of OperatorNode with arity 0
    node = cgp.ConstantFloat(idx, addresses)
    node_repr = str(node)
    assert node_repr == "ConstantFloat(idx: 0, active: False, arity: 0, addresses of inputs []"

    # Test OutputNode
    node = cgp.node_input_output.OutputNode(idx, addresses)
    node_repr = str(node)
    assert node_repr == "OutputNode(idx: 0, active: False, arity: 1, addresses of inputs [1]"

    # Test example of OperatorNode with arity 2
    node = cgp.Add(idx, addresses)
    node_repr = str(node)
    assert node_repr == "Add(idx: 0, active: False, arity: 2, addresses of inputs [1, 2]"


def test_custom_node():
    class MyScaledAdd(cgp.node.OperatorNode):

        _arity = 2
        _def_output = "2.0 * (x_0 + x_1)"

    primitives = (MyScaledAdd,)
    params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
        "levels_back": 1,
        "primitives": primitives,
    }

    genome = cgp.Genome(**params)

    # f(x) = 2 * (x[0] + x[1])
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]

    x = [5.0, 1.5]
    y_target = 2 * (x[0] + x[1])

    _test_to_x_compilations(genome, x, y_target)


def test_custom_node_with_custom_atomic_operator():
    @cgp.atomic_operator
    def f_scale(x):
        return 2.0 * x

    class MyScaledAdd(cgp.node.OperatorNode):

        _arity = 2
        _def_output = "f_scale((x_0 + x_1))"

    primitives = (MyScaledAdd,)
    params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
        "levels_back": 1,
        "primitives": primitives,
    }

    genome = cgp.Genome(**params)

    # f(x) = f_scale(x[0] + x[1])
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]

    x = [5.0, 1.5]
    y_target = 2 * (x[0] + x[1])

    _test_to_x_compilations(genome, x, y_target, test_to_sympy=False)


def test_custom_node_with_custom_atomic_operator_with_external_library():
    scipy_const = pytest.importorskip("scipy.constants")

    @cgp.atomic_operator
    def f_scale(x):
        return scipy_const.golden_ratio * x

    class MyScaledAdd(cgp.node.OperatorNode):

        _arity = 2
        _def_output = "f_scale((x_0 + x_1))"

    primitives = (MyScaledAdd,)
    params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
        "levels_back": 1,
        "primitives": primitives,
    }

    genome = cgp.Genome(**params)

    # f(x) = f_scale(x[0] + x[1])
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]

    x = [5.0, 1.5]
    y_target = scipy_const.golden_ratio * (x[0] + x[1])

    _test_to_x_compilations(genome, x, y_target, test_to_sympy=False)
