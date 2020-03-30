import numpy as np
import pytest
import sympy
import torch
from itertools import product

import gp


SEED = np.random.randint(2 ** 31)


def test_direct_input_output():
    params = {"n_inputs": 1, "n_outputs": 1, "n_columns": 3, "n_rows": 3, "levels_back": 2}
    primitives = [gp.CGPAdd, gp.CGPSub]
    genome = gp.CGPGenome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.randomize(np.random)

    genome[-2:] = [0, None]  # set inputs for output node to input node
    graph = gp.CGPGraph(genome)

    x = [2.14159]
    y = graph(x)

    assert abs(x[0] - y[0]) < 1e-15


def test_to_func_simple():
    primitives = [gp.CGPAdd]
    genome = gp.CGPGenome(2, 1, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert abs(x[0] + x[1] - y[0]) < 1e-15

    primitives = [gp.CGPSub]
    genome = gp.CGPGenome(2, 1, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert abs(x[0] - x[1] - y[0]) < 1e-15


def test_compile_two_columns():
    primitives = [gp.CGPAdd, gp.CGPSub]
    genome = gp.CGPGenome(2, 1, 2, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 1, 0, 2, -2, 3, None]
    graph = gp.CGPGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert abs(x[0] - (x[0] + x[1]) - y[0]) < 1e-15


def test_compile_two_columns_two_rows():
    primitives = [gp.CGPAdd, gp.CGPSub]
    genome = gp.CGPGenome(2, 2, 2, 2, 1, primitives)

    genome.dna = [
        -1,
        None,
        None,
        -1,
        None,
        None,
        0,
        0,
        1,
        1,
        0,
        1,
        0,
        0,
        2,
        0,
        2,
        3,
        -2,
        4,
        None,
        -2,
        5,
        None,
    ]
    graph = gp.CGPGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert abs(x[0] + (x[0] + x[1]) - y[0]) < 1e-15
    assert abs((x[0] + x[1]) + (x[0] - x[1]) - y[1]) < 1e-15


def test_compile_addsubmul():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 2, "n_rows": 2, "levels_back": 1}

    primitives = [gp.CGPAdd, gp.CGPSub, gp.CGPMul]
    genome = gp.CGPGenome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.dna = [-1, None, None, -1, None, None, 2, 0, 1, 1, 0, 1, 1, 2, 3, 0, 0, 0, -2, 4, None]
    graph = gp.CGPGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert abs(((x[0] * x[1]) - (x[0] - x[1])) - y[0]) < 1e-15


batch_sizes = [1, 10]
primitives = [gp.CGPMul, gp.CGPConstantFloat]
genomes = [gp.CGPGenome(1, 1, 2, 2, 1, primitives) for i in range(2)]
# Function: f(x) = 1*x
genomes[0].dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 3, None]
# Function: f(x) = 1
genomes[1].dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 1, None]

genomes += [gp.CGPGenome(1, 2, 2, 2, 1, primitives) for i in range(2)]
# Function: f(x) = (1*x, 1*1)
genomes[2].dna = [
    -1,
    None,
    None,
    1,
    None,
    None,
    1,
    None,
    None,
    0,
    0,
    1,
    0,
    1,
    1,
    -2,
    3,
    None,
    -2,
    4,
    None,
]
# Function: f(x) = (1, x*x)
genomes[3].dna = [
    -1,
    None,
    None,
    1,
    None,
    None,
    1,
    None,
    None,
    0,
    1,
    1,
    0,
    0,
    1,
    -2,
    1,
    None,
    -2,
    3,
    None,
]


@pytest.mark.parametrize("genome, batch_size", product(genomes, batch_sizes))
def test_compile_torch_output_shape(genome, batch_size):
    graph = gp.CGPGraph(genome)
    c = graph.to_torch()
    x = torch.Tensor(batch_size, 1).normal_()
    y = c(x)
    assert y.shape == (batch_size, genome._n_outputs)


def test_to_sympy():
    primitives = [gp.CGPAdd, gp.CGPConstantFloat]
    genome = gp.CGPGenome(1, 1, 2, 2, 1, primitives)

    genome.dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 3, None]
    graph = gp.CGPGraph(genome)

    x_0_target, y_0_target = sympy.symbols("x_0_target y_0_target")
    y_0_target = x_0_target + 1.0

    y_0 = graph.to_sympy()[0]

    for x in np.random.normal(size=100):
        assert abs(y_0_target.subs("x_0_target", x).evalf() - y_0.subs("x_0", x).evalf()) < 1e-12


def test_catch_invalid_sympy_expr():
    primitives = [gp.CGPSub, gp.CGPDiv]
    genome = gp.CGPGenome(1, 1, 2, 1, 1, primitives)

    # x[0] / (x[0] - x[0])
    genome.dna = [-1, None, None, 0, 0, 0, 1, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    with pytest.raises(gp.exceptions.InvalidSympyExpression):
        graph.to_sympy(simplify=True)


def test_allow_powers_of_x_0():
    primitives = [gp.CGPSub, gp.CGPMul]
    genome = gp.CGPGenome(1, 1, 2, 1, 1, primitives)

    # x[0] ** 2
    genome.dna = [-1, None, None, 0, 0, 0, 1, 0, 0, -2, 2, None]
    graph = gp.CGPGraph(genome)
    graph.to_sympy(simplify=True)
