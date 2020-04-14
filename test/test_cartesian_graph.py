import numpy as np
import pytest
import sympy
import torch
from itertools import product

import gp


SEED = np.random.randint(2 ** 31)


def test_direct_input_output():
    params = {"n_inputs": 1, "n_outputs": 1, "n_columns": 3, "n_rows": 3, "levels_back": 2}
    primitives = [gp.Add, gp.Sub]
    genome = gp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.randomize(np.random)

    genome[-2:] = [0, None]  # set inputs for output node to input node
    graph = gp.CartesianGraph(genome)

    x = [2.14159]
    y = graph(x)

    assert pytest.approx(x[0] == y[0])


def test_to_func_simple():
    primitives = [gp.Add]
    genome = gp.Genome(2, 1, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert pytest.approx(x[0] + x[1] == y[0])

    primitives = [gp.Sub]
    genome = gp.Genome(2, 1, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert pytest.approx(x[0] - x[1] == y[0])


def test_compile_two_columns():
    primitives = [gp.Add, gp.Sub]
    genome = gp.Genome(2, 1, 2, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 1, 0, 2, -2, 3, None]
    graph = gp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert pytest.approx(x[0] - (x[0] + x[1]) == y[0])


def test_compile_two_columns_two_rows():
    primitives = [gp.Add, gp.Sub]
    genome = gp.Genome(2, 2, 2, 2, 1, primitives)

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
    graph = gp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert pytest.approx(x[0] + (x[0] + x[1]) == y[0])
    assert pytest.approx((x[0] + x[1]) + (x[0] - x[1]) == y[1])


def test_compile_addsubmul():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 2, "n_rows": 2, "levels_back": 1}

    primitives = [gp.Add, gp.Sub, gp.Mul]
    genome = gp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.dna = [-1, None, None, -1, None, None, 2, 0, 1, 1, 0, 1, 1, 2, 3, 0, 0, 0, -2, 4, None]
    graph = gp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert pytest.approx(((x[0] * x[1]) - (x[0] - x[1])) == y[0])


def test_to_torch_and_backprop():
    primitives = [gp.Mul, gp.ConstantFloat]
    genome = gp.Genome(1, 1, 2, 2, 1, primitives)
    genome.dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 3, None]
    graph = gp.CartesianGraph(genome)

    c = graph.to_torch()

    optimizer = torch.optim.SGD(c.parameters(), lr=1e-1)
    criterion = torch.nn.MSELoss()

    for i in range(200):

        x = torch.Tensor(1, 1).normal_()
        y = c(x)

        y_target = -2.14159 * x

        loss = criterion(y, y_target)
        c.zero_grad()
        loss.backward()

        optimizer.step()
    assert pytest.approx(float(loss.detach()) == 0.0)

    x = [3.0]
    x_torch = torch.Tensor(x).view(1, 1)
    assert pytest.approx(c(x_torch)[0].detach().numpy() != graph(x))
    graph.update_parameters_from_torch_class(c)
    assert pytest.approx(c(x_torch)[0].detach().numpy() == graph(x))


batch_sizes = [1, 10]
primitives = [gp.Mul, gp.ConstantFloat]
genomes = [gp.Genome(1, 1, 2, 2, 1, primitives) for i in range(2)]
# Function: f(x) = 1*x
genomes[0].dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 3, None]
# Function: f(x) = 1
genomes[1].dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 1, None]

genomes += [gp.Genome(1, 2, 2, 2, 1, primitives) for i in range(2)]
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
    graph = gp.CartesianGraph(genome)
    c = graph.to_torch()
    x = torch.Tensor(batch_size, 1).normal_()
    y = c(x)
    assert y.shape == (batch_size, genome._n_outputs)


def test_to_sympy():
    primitives = [gp.Add, gp.ConstantFloat]
    genome = gp.Genome(1, 1, 2, 2, 1, primitives)

    genome.dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 3, None]
    graph = gp.CartesianGraph(genome)

    x_0_target, y_0_target = sympy.symbols("x_0_target y_0_target")
    y_0_target = x_0_target + 1.0

    y_0 = graph.to_sympy()[0]

    for x in np.random.normal(size=100):
        assert pytest.approx(
            y_0_target.subs("x_0_target", x).evalf() == y_0.subs("x_0", x).evalf()
        )


def test_catch_invalid_sympy_expr():
    primitives = [gp.Sub, gp.Div]
    genome = gp.Genome(1, 1, 2, 1, 1, primitives)

    # x[0] / (x[0] - x[0])
    genome.dna = [-1, None, None, 0, 0, 0, 1, 0, 1, -2, 2, None]
    graph = gp.CartesianGraph(genome)

    with pytest.raises(gp.exceptions.InvalidSympyExpression):
        graph.to_sympy(simplify=True)


def test_allow_powers_of_x_0():
    primitives = [gp.Sub, gp.Mul]
    genome = gp.Genome(1, 1, 2, 1, 1, primitives)

    # x[0] ** 2
    genome.dna = [-1, None, None, 0, 0, 0, 1, 0, 0, -2, 2, None]
    graph = gp.CartesianGraph(genome)
    graph.to_sympy(simplify=True)


def test_input_dim_python():
    rng = np.random.RandomState(SEED)

    genome = gp.Genome(2, 1, 1, 1, 1, [gp.ConstantFloat])
    genome.randomize(rng)
    f = gp.CartesianGraph(genome).to_func()

    # fail for too short input
    with pytest.raises(ValueError):
        f([None])

    # fail for too long input
    with pytest.raises(ValueError):
        f([None, None, None])

    # do not fail for input with correct length
    f([None, None])


def test_input_dim_torch():
    rng = np.random.RandomState(SEED)

    genome = gp.Genome(2, 1, 1, 1, 1, [gp.ConstantFloat])
    genome.randomize(rng)
    f = gp.CartesianGraph(genome).to_torch()

    # fail for missing batch dimension
    with pytest.raises(ValueError):
        f(torch.Tensor([1.0]))

    # fail for too short input
    with pytest.raises(ValueError):
        f(torch.Tensor([1.0]).reshape(-1, 1))

    # fail for too long input
    with pytest.raises(ValueError):
        f(torch.Tensor([1.0, 1.0, 1.0]).reshape(-1, 3))

    # do not fail for input with correct shape
    f(torch.Tensor([1.0, 1.0]).reshape(-1, 2))


def test_pretty_str():
    primitives = [gp.Sub, gp.Mul]
    genome = gp.Genome(1, 1, 2, 1, 1, primitives)

    # x[0] ** 2
    genome.dna = [-1, None, None, 0, 0, 0, 1, 0, 0, -2, 2, None]
    graph = gp.CartesianGraph(genome)

    pretty_str = graph.pretty_str()

    for node in graph.input_nodes:
        assert node.__class__.__name__ in pretty_str
    for node in graph.output_nodes:
        assert node.__class__.__name__ in pretty_str
    for node in graph.hidden_nodes:
        assert node.__class__.__name__ in pretty_str
