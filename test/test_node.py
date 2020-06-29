import math
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
    graph = cgp.CartesianGraph(genome)

    x = [5.0, 1.5]
    y = graph(x)

    assert x[0] + x[1] == pytest.approx(y[0])


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
    graph = cgp.CartesianGraph(genome)

    x = [5.0, 1.5]
    y = graph(x)

    assert x[0] - x[1] == pytest.approx(y[0])


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
    graph = cgp.CartesianGraph(genome)

    x = [5.0, 1.5]
    y = graph(x)

    assert x[0] * x[1] == pytest.approx(y[0])


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
    graph = cgp.CartesianGraph(genome)

    x = [5.0, 1.5]
    y = graph(x)

    assert x[0] / x[1] == pytest.approx(y[0])


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
    graph = cgp.CartesianGraph(genome)

    x = [5.0, 1.5]
    y = graph(x)

    assert x[0] ** x[1] == pytest.approx(y[0])


def test_constant_float():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.ConstantFloat,)
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
    graph = cgp.CartesianGraph(genome)

    x = [None, None]
    y = graph(x)

    # by default the output value of the ConstantFloat node is 1.0
    assert 1.0 == pytest.approx(y[0])


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
    f = cgp.CartesianGraph(genome).to_func()
    y = f([0.0])[0]

    assert y == pytest.approx(1.0)


def test_parameter_w_custom_initial_value():
    initial_value = 3.1415

    class CustomParameter(cgp.Parameter):
        @staticmethod
        def initial_value(_):
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
    f = cgp.CartesianGraph(genome).to_func()
    y = f([0.0])[0]

    assert y == pytest.approx(initial_value)


def test_parameter_w_random_initial_value(rng_seed):
    np.random.seed(rng_seed)

    min_val = 0.5
    max_val = 1.5

    class CustomParameter(cgp.Parameter):
        @staticmethod
        def initial_value(_):
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


def test_parametrized_add():
    torch = pytest.importorskip("torch")

    primitives = (cgp.ParametrizedAdd,)
    genome = cgp.Genome(2, 1, 1, 1, 1, primitives)
    # f(x) = w * (x[0] + x[1]) + b
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

    w = math.pi
    b = math.e

    assert genome._parameter_names_to_values["<w2>"] == pytest.approx(1.0)
    assert genome._parameter_names_to_values["<b2>"] == pytest.approx(0.0)

    genome._parameter_names_to_values["<w2>"] = w
    genome._parameter_names_to_values["<b2>"] = b

    graph = cgp.CartesianGraph(genome)

    x = [1.5, 2.5]
    y_target = w * (x[0] + x[1]) + b

    # func
    f = graph.to_func()
    assert f(x)[0] == pytest.approx(y_target)

    # numpy
    f = graph.to_numpy()
    assert f(np.array(x).reshape(1, -1))[0, 0] == pytest.approx(y_target)

    # TODO the two tests below should be in seperate function such
    # that they can be skipped if the respective module is not
    # available

    # sympy
    f = graph.to_sympy()[0]
    assert f.subs({"x_0": x[0], "x_1": x[1]}).evalf() == pytest.approx(y_target)

    # torch
    f = graph.to_torch()
    assert f(torch.Tensor(x).reshape(1, -1))[0, 0] == pytest.approx(y_target)
