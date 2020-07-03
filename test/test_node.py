import numpy as np
import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE


def test_inputs_are_cut_to_match_arity():
    """Test that even if a list of inputs longer than the node arity is
    provided, Node.inputs only returns the initial <arity> inputs,
    ignoring the inactive genes.

    """
    idx = 0
    inputs = [1, 2, 3, 4]

    node = cgp.ConstantFloat(idx, inputs)
    assert node.inputs == []

    node = cgp.node.OutputNode(idx, inputs)
    assert node.inputs == inputs[:1]

    node = cgp.Add(idx, inputs)
    assert node.inputs == inputs[:2]


def _test_graph_call_and_to_x_compilations(
    genome,
    x,
    y_target,
    *,
    test_graph_call=True,
    test_to_func=True,
    test_to_numpy=True,
    test_to_torch=True,
    test_to_sympy=True,
):
    if test_graph_call:
        _test_graph_call(genome, x, y_target)
    if test_to_func:
        _test_to_func(genome, x, y_target)
    if test_to_numpy:
        _test_to_numpy(genome, x, y_target)
    if test_to_torch:
        _test_to_torch(genome, x, y_target)
    if test_to_sympy:
        _test_to_sympy(genome, x, y_target)


def _test_graph_call(genome, x, y_target):
    graph = cgp.CartesianGraph(genome)
    assert graph(x) == pytest.approx(y_target)


def _test_to_func(genome, x, y_target):
    graph = cgp.CartesianGraph(genome)
    assert graph.to_func()(x) == pytest.approx(y_target)


def _test_to_numpy(genome, x, y_target):
    graph = cgp.CartesianGraph(genome)
    assert graph.to_numpy()(np.array(x).reshape(1, -1)) == pytest.approx(y_target)


def _test_to_torch(genome, x, y_target):
    torch = pytest.importorskip("torch")
    graph = cgp.CartesianGraph(genome)
    assert graph.to_numpy()(torch.Tensor(x).reshape(1, -1)) == pytest.approx(y_target)


def _test_to_sympy(genome, x, y_target):
    pytest.importorskip("sympy")
    graph = cgp.CartesianGraph(genome)
    assert [graph.to_sympy()[0].subs({f"x_{i}": x[i] for i in range(len(x))})] == pytest.approx(
        y_target
    )


def test_add():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
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
    y_target = [x[0] + x[1]]

    _test_graph_call_and_to_x_compilations(genome, x, y_target)


def test_sub():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Sub,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
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
    y_target = [x[0] - x[1]]

    _test_graph_call_and_to_x_compilations(genome, x, y_target)


def test_mul():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Mul,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
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
    y_target = [x[0] * x[1]]

    _test_graph_call_and_to_x_compilations(genome, x, y_target)


def test_div():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Div,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
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
    y_target = [x[0] / x[1]]

    _test_graph_call_and_to_x_compilations(genome, x, y_target)


def test_pow():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Pow,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
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
    y_target = [x[0] ** x[1]]

    _test_graph_call_and_to_x_compilations(genome, x, y_target)


def test_constant_float():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.ConstantFloat,)
    # f(x) = c
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
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
    y_target = [1.0]  # by default the output value of the ConstantFloat node is 1.0

    _test_graph_call_and_to_x_compilations(genome, x, y_target)


def test_parameter():
    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
        "levels_back": None,
    }
    primitives = (cgp.Parameter,)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = c
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        0,
        0,
        ID_OUTPUT_NODE,
        1,
    ]

    x = [1.0]
    y_target = [1.0]  # by default the output value of the Parameter node is 1.0

    _test_graph_call_and_to_x_compilations(genome, x, y_target, test_graph_call=False)


def test_parameter_w_custom_initial_value():
    initial_value = 3.1415

    class CustomParameter(cgp.Parameter):
        @staticmethod
        def initial_value():
            return initial_value

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
        "levels_back": None,
    }
    primitives = (CustomParameter,)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = c
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        0,
        0,
        ID_OUTPUT_NODE,
        1,
    ]

    x = [1.0]
    y_target = [initial_value]

    _test_graph_call_and_to_x_compilations(genome, x, y_target, test_graph_call=False)


def test_parameter_w_random_initial_value(rng_seed):
    np.random.seed(rng_seed)

    min_val = 0.5
    max_val = 1.5

    class CustomParameter(cgp.Parameter):
        @staticmethod
        def initial_value():
            return np.random.uniform(min_val, max_val)

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
        "levels_back": None,
    }
    primitives = (CustomParameter,)
    genome = cgp.Genome(**genome_params, primitives=primitives)
    # f(x) = c
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        0,
        0,
        ID_OUTPUT_NODE,
        1,
    ]
    f = cgp.CartesianGraph(genome).to_func()
    y = f([0.0])[0]

    assert min_val <= y
    assert y <= max_val
    assert y != pytest.approx(1.0)
