import itertools
import numpy as np
import pytest

import gp


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

    assert x[0] == pytest.approx(y[0])


def test_to_func_simple():
    primitives = [gp.Add]
    genome = gp.Genome(2, 1, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert x[0] + x[1] == pytest.approx(y[0])

    primitives = [gp.Sub]
    genome = gp.Genome(2, 1, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert x[0] - x[1] == pytest.approx(y[0])


def test_compile_two_columns():
    primitives = [gp.Add, gp.Sub]
    genome = gp.Genome(2, 1, 2, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 1, 0, 2, -2, 3, None]
    graph = gp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert x[0] - (x[0] + x[1]) == pytest.approx(y[0])


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

    assert x[0] + (x[0] + x[1]) == pytest.approx(y[0])
    assert (x[0] + x[1]) + (x[0] - x[1]) == pytest.approx(y[1])


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

    assert (x[0] * x[1]) - (x[0] - x[1]) == pytest.approx(y[0])


def test_to_numpy():
    primitives = [gp.Add, gp.Mul, gp.ConstantFloat]
    genome = gp.Genome(1, 1, 2, 2, 1, primitives)
    # f(x) = x ** 2 + 1.
    genome.dna = [-1, None, None, 2, 0, 0, 1, 0, 0, 0, 1, 2, 0, 0, 1, -2, 3, None]
    graph = gp.CartesianGraph(genome)
    f = graph.to_numpy()

    x = np.random.normal(size=(100, 1))
    y = f(x)
    y_target = x ** 2 + 1.0

    assert y == pytest.approx(y_target)


batch_sizes = [1, 10]
primitives = [gp.Mul, gp.ConstantFloat]
genomes = [gp.Genome(1, 1, 2, 2, 1, primitives) for i in range(2)]
# Function: f(x) = 1*x
genomes[0].dna = [-1, None, None, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, -2, 3, None]
# Function: f(x) = 1
genomes[1].dna = [-1, None, None, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, -2, 1, None]

genomes += [gp.Genome(1, 2, 2, 2, 1, primitives) for i in range(2)]
# Function: f(x) = (1*x, 1*1)
genomes[2].dna = [
    -1,
    None,
    None,
    1,
    0,
    0,
    1,
    0,
    0,
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
    0,
    0,
    1,
    0,
    0,
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


@pytest.mark.parametrize("genome, batch_size", itertools.product(genomes, batch_sizes))
def test_compile_numpy_output_shape(genome, batch_size):

    c = gp.CartesianGraph(genome).to_numpy()
    x = np.random.normal(size=(batch_size, 1))
    y = c(x)
    assert y.shape == (batch_size, genome._n_outputs)


@pytest.mark.parametrize("genome, batch_size", itertools.product(genomes, batch_sizes))
def test_compile_torch_output_shape(genome, batch_size):
    torch = pytest.importorskip("torch")

    c = gp.CartesianGraph(genome).to_torch()
    x = torch.Tensor(batch_size, 1).normal_()
    y = c(x)
    assert y.shape == (batch_size, genome._n_outputs)


def test_to_sympy():
    sympy = pytest.importorskip("sympy")

    primitives = [gp.Add, gp.ConstantFloat]
    genome = gp.Genome(1, 1, 2, 2, 1, primitives)

    genome.dna = [-1, None, None, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, -2, 3, None]
    graph = gp.CartesianGraph(genome)

    x_0_target, y_0_target = sympy.symbols("x_0_target y_0_target")
    y_0_target = x_0_target + 1.0

    y_0 = graph.to_sympy()[0]

    for x in np.random.normal(size=100):
        assert pytest.approx(
            y_0_target.subs("x_0_target", x).evalf() == y_0.subs("x_0", x).evalf()
        )


def test_catch_invalid_sympy_expr():
    pytest.importorskip("sympy")

    primitives = [gp.Sub, gp.Div]
    genome = gp.Genome(1, 1, 2, 1, 1, primitives)

    # x[0] / (x[0] - x[0])
    genome.dna = [-1, None, None, 0, 0, 0, 1, 0, 1, -2, 2, None]
    graph = gp.CartesianGraph(genome)

    with pytest.raises(Exception):
        graph.to_sympy(simplify=True)


def test_allow_powers_of_x_0():
    pytest.importorskip("sympy")

    primitives = [gp.Sub, gp.Mul]
    genome = gp.Genome(1, 1, 2, 1, 1, primitives)

    # x[0] ** 2
    genome.dna = [-1, None, None, 0, 0, 0, 1, 0, 0, -2, 2, None]
    graph = gp.CartesianGraph(genome)
    graph.to_sympy(simplify=True)


def test_input_dim_python(rng_seed):
    rng = np.random.RandomState(rng_seed)

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


def test_input_dim_numpy(rng_seed):
    rng = np.random.RandomState(rng_seed)

    genome = gp.Genome(2, 1, 1, 1, 1, [gp.ConstantFloat])
    genome.randomize(rng)
    f = gp.CartesianGraph(genome).to_numpy()

    # fail for missing batch dimension
    with pytest.raises(ValueError):
        f(np.array([1.0]))

    # fail for too short input
    with pytest.raises(ValueError):
        f(np.array([1.0]).reshape(-1, 1))

    # fail for too long input
    with pytest.raises(ValueError):
        f(np.array([1.0, 1.0, 1.0]).reshape(-1, 3))

    # do not fail for input with correct shape
    f(np.array([1.0, 1.0]).reshape(-1, 2))


def test_input_dim_torch(rng_seed):
    torch = pytest.importorskip("torch")

    rng = np.random.RandomState(rng_seed)

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


def test_pretty_str_with_unequal_inputs_rows_outputs():
    primitives = [gp.Add]

    # less rows than inputs/outputs
    genome = gp.Genome(1, 1, 1, 2, 1, primitives)
    # f(x) = x[0] + x[0]
    genome.dna = [-1, None, None, 0, 0, 0, 0, 0, 0, -2, 1, None]
    graph = gp.CartesianGraph(genome)

    expected_pretty_str = """
00 * InputNode          \t01 * Add (00,00)        \t03 * OutputNode (01)    \t
                        \t02   Add                \t                        \t
"""
    assert graph.pretty_str() == expected_pretty_str

    # more rows than inputs/outputs
    genome = gp.Genome(3, 3, 1, 2, 1, primitives)
    # f(x) = [x[0] + x[1], x[0] + x[1], x[1] + x[2]]
    genome.dna = [
        -1,
        None,
        None,
        -1,
        None,
        None,
        -1,
        None,
        None,
        0,
        0,
        1,
        0,
        1,
        2,
        -2,
        3,
        None,
        -2,
        3,
        None,
        -2,
        4,
        None,
    ]
    graph = gp.CartesianGraph(genome)

    expected_pretty_str = """
00 * InputNode          \t03 * Add (00,01)        \t05 * OutputNode (03)    \t
01 * InputNode          \t04 * Add (01,02)        \t06 * OutputNode (03)    \t
02 * InputNode          \t                        \t07 * OutputNode (04)    \t
"""
    assert graph.pretty_str() == expected_pretty_str
