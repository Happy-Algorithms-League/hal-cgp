import itertools
import numpy as np
import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE


def test_direct_input_output():
    params = {"n_inputs": 1, "n_outputs": 1, "n_columns": 3, "n_rows": 3, "levels_back": 2}
    primitives = (cgp.Add, cgp.Sub)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.randomize(np.random)

    genome[-2:] = [0, ID_NON_CODING_GENE]  # set inputs for output node to input node
    graph = cgp.CartesianGraph(genome)

    x = [2.14159]
    y = graph(x)

    assert x[0] == pytest.approx(y[0])


def test_to_func_simple():
    primitives = (cgp.Add,)
    genome = cgp.Genome(2, 1, 1, 1, 1, primitives)

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
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert x[0] + x[1] == pytest.approx(y[0])

    primitives = (cgp.Sub,)
    genome = cgp.Genome(2, 1, 1, 1, 1, primitives)

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
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert x[0] - x[1] == pytest.approx(y[0])


def test_compile_two_columns():
    primitives = (cgp.Add, cgp.Sub)
    genome = cgp.Genome(2, 1, 2, 1, 1, primitives)

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
        1,
        0,
        2,
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert x[0] - (x[0] + x[1]) == pytest.approx(y[0])


def test_compile_two_columns_two_rows():
    primitives = (cgp.Add, cgp.Sub)
    genome = cgp.Genome(2, 2, 2, 2, 1, primitives)

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
        1,
        0,
        1,
        0,
        0,
        2,
        0,
        2,
        3,
        ID_OUTPUT_NODE,
        4,
        ID_NON_CODING_GENE,
        ID_OUTPUT_NODE,
        5,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert x[0] + (x[0] + x[1]) == pytest.approx(y[0])
    assert (x[0] + x[1]) + (x[0] - x[1]) == pytest.approx(y[1])


def test_compile_addsubmul():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 2, "n_rows": 2, "levels_back": 1}

    primitives = (cgp.Add, cgp.Sub, cgp.Mul)
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
        2,
        0,
        1,
        1,
        0,
        1,
        1,
        2,
        3,
        0,
        0,
        0,
        ID_OUTPUT_NODE,
        4,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)
    f = graph.to_func()

    x = [5.0, 2.0]
    y = f(x)

    assert (x[0] * x[1]) - (x[0] - x[1]) == pytest.approx(y[0])


def test_to_numpy():
    primitives = (cgp.Add, cgp.Mul, cgp.ConstantFloat)
    genome = cgp.Genome(1, 1, 2, 2, 1, primitives)
    # f(x) = x ** 2 + 1.
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        2,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        2,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)
    f = graph.to_numpy()

    x = np.random.normal(size=(100, 1))
    y = f(x)
    y_target = x ** 2 + 1.0

    assert y == pytest.approx(y_target)


batch_sizes = [1, 10]
primitives = (cgp.Mul, cgp.ConstantFloat)
genome = [cgp.Genome(1, 1, 2, 2, 1, primitives) for i in range(2)]
# Function: f(x) = 1*x
genome[0].dna = [
    ID_INPUT_NODE,
    ID_NON_CODING_GENE,
    ID_NON_CODING_GENE,
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
    0,
    1,
    ID_OUTPUT_NODE,
    3,
    ID_NON_CODING_GENE,
]
# Function: f(x) = 1
genome[1].dna = [
    ID_INPUT_NODE,
    ID_NON_CODING_GENE,
    ID_NON_CODING_GENE,
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
    0,
    1,
    ID_OUTPUT_NODE,
    1,
    ID_NON_CODING_GENE,
]

genome += [cgp.Genome(1, 2, 2, 2, 1, primitives) for i in range(2)]
# Function: f(x) = (1*x, 1*1)
genome[2].dna = [
    ID_INPUT_NODE,
    ID_NON_CODING_GENE,
    ID_NON_CODING_GENE,
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
    ID_OUTPUT_NODE,
    3,
    ID_NON_CODING_GENE,
    ID_OUTPUT_NODE,
    4,
    ID_NON_CODING_GENE,
]
# Function: f(x) = (1, x*x)
genome[3].dna = [
    ID_INPUT_NODE,
    ID_NON_CODING_GENE,
    ID_NON_CODING_GENE,
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
    ID_OUTPUT_NODE,
    1,
    ID_NON_CODING_GENE,
    ID_OUTPUT_NODE,
    3,
    ID_NON_CODING_GENE,
]


@pytest.mark.parametrize("genome, batch_size", itertools.product(genome, batch_sizes))
def test_compile_numpy_output_shape(genome, batch_size):

    c = cgp.CartesianGraph(genome).to_numpy()
    x = np.random.normal(size=(batch_size, 1))
    y = c(x)
    assert y.shape == (batch_size, genome._n_outputs)


@pytest.mark.parametrize("genome, batch_size", itertools.product(genome, batch_sizes))
def test_compile_torch_output_shape(genome, batch_size):
    torch = pytest.importorskip("torch")

    c = cgp.CartesianGraph(genome).to_torch()
    x = torch.Tensor(batch_size, 1).normal_()
    y = c(x)
    assert y.shape == (batch_size, genome._n_outputs)


def test_to_sympy():
    sympy = pytest.importorskip("sympy")

    primitives = (cgp.Add, cgp.ConstantFloat)
    genome = cgp.Genome(1, 1, 2, 2, 1, primitives)

    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
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
        0,
        1,
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)

    x_0_target, y_0_target = sympy.symbols("x_0_target y_0_target")
    y_0_target = x_0_target + 1.0

    y_0 = graph.to_sympy()[0]

    for x in np.random.normal(size=100):
        assert pytest.approx(
            y_0_target.subs("x_0_target", x).evalf() == y_0.subs("x_0", x).evalf()
        )


def test_allow_sympy_expr_with_infinities():
    pytest.importorskip("sympy")

    primitives = (cgp.Sub, cgp.Div)
    genome = cgp.Genome(1, 1, 2, 1, 1, primitives)

    # x[0] / (x[0] - x[0])
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        0,
        1,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)

    expr = graph.to_sympy(simplify=True)[0]
    # complex infinity should appear in expression
    assert "zoo" in str(expr)


def test_allow_powers_of_x_0():
    pytest.importorskip("sympy")

    primitives = (cgp.Sub, cgp.Mul)
    genome = cgp.Genome(1, 1, 2, 1, 1, primitives)

    # x[0] ** 2
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        0,
        1,
        0,
        0,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)
    graph.to_sympy(simplify=True)


def test_input_dim_python(rng_seed):
    rng = np.random.RandomState(rng_seed)

    genome = cgp.Genome(2, 1, 1, 1, 1, (cgp.ConstantFloat,))
    genome.randomize(rng)
    f = cgp.CartesianGraph(genome).to_func()

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

    genome = cgp.Genome(2, 1, 1, 1, 1, (cgp.ConstantFloat,))
    genome.randomize(rng)
    f = cgp.CartesianGraph(genome).to_numpy()

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

    genome = cgp.Genome(2, 1, 1, 1, 1, (cgp.ConstantFloat,))
    genome.randomize(rng)
    f = cgp.CartesianGraph(genome).to_torch()

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
    primitives = (cgp.Sub, cgp.Mul)
    genome = cgp.Genome(1, 1, 2, 1, 1, primitives)

    # x[0] ** 2
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        0,
        1,
        0,
        0,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)

    pretty_str = graph.pretty_str()

    for node in graph.input_nodes:
        assert node.__class__.__name__ in pretty_str
    for node in graph.output_nodes:
        assert node.__class__.__name__ in pretty_str
    for node in graph.hidden_nodes:
        assert node.__class__.__name__ in pretty_str


def test_pretty_str_with_unequal_inputs_rows_outputs():
    primitives = (cgp.Add,)

    # less rows than inputs/outputs
    genome = cgp.Genome(1, 1, 1, 2, 1, primitives)
    # f(x) = x[0] + x[0]
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
        ID_OUTPUT_NODE,
        1,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)

    expected_pretty_str = """
00 * InputNode          \t01 * Add (00,00)        \t03 * OutputNode (01)    \t
                        \t02   Add                \t                        \t
"""
    assert graph.pretty_str() == expected_pretty_str

    # more rows than inputs/outputs
    genome = cgp.Genome(3, 3, 1, 2, 1, primitives)
    # f(x) = [x[0] + x[1], x[0] + x[1], x[1] + x[2]]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        0,
        1,
        2,
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
        ID_OUTPUT_NODE,
        4,
        ID_NON_CODING_GENE,
    ]
    graph = cgp.CartesianGraph(genome)

    expected_pretty_str = """
00 * InputNode          \t03 * Add (00,01)        \t05 * OutputNode (03)    \t
01 * InputNode          \t04 * Add (01,02)        \t06 * OutputNode (03)    \t
02 * InputNode          \t                        \t07 * OutputNode (04)    \t
"""
    assert graph.pretty_str() == expected_pretty_str
