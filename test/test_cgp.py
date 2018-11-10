import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys
import torch

sys.path.insert(0, '../')
import gp


# -> genome
def test_permissable_inputs():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 4,
        'n_rows': 3,
        'levels_back': 2,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)

    expected = [
        [0, 1],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 5, 6, 7, 8, 9, 10],
    ]

    for hidden_column_idx in range(params['n_columns']):
        assert(expected[hidden_column_idx] == genome._permissable_inputs(hidden_column_idx, params['levels_back']))


# -> genome
def test_region_generators():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 0, None]

    for region in genome.input_regions():
        assert(region == [-1, None, None])

    for region in genome.hidden_regions():
        assert(region == [0, 0, 1])

    for region in genome.output_regions():
        assert(region == [-2, 0, None])


# -> genome
def test_check_dna_consistency():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 0, None]

    # invalid length
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, -2, -1, -2, 0, None, 0]

    # invalid function gene for input node
    with pytest.raises(ValueError):
        genome.dna = [0, None, None, -1, None, None, 0, -2, 0, -2, 0, None]

    # invalid input gene for input node
    with pytest.raises(ValueError):
        genome.dna = [-1, 0, None, -1, None, None, 0, -2, 0, -2, 0, None]

    # invalid function gene for hidden node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 2, 0, 1, -2, 0, None]

    # invalid input gene for hidden node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, 2, 1, -2, 0, None]

    # invalid function gene for output node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 0, 0, None]

    # invalid input gene for input node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 3, None]

    # invalid non-coding input gene for output node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 0, 0]


# -> node
def test_add():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert(abs(x[0] + x[1] - y[0]) < 1e-15)


# -> node
def test_sub():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPSub])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert(abs(x[0] - x[1] - y[0]) < 1e-15)


# -> node
def test_mul():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPMul])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert(abs((x[0] * x[1]) - y[0]) < 1e-15)


# -> graph
def test_direct_input_output():
    params = {
        'n_inputs': 1,
        'n_outputs': 1,
        'n_columns': 3,
        'n_rows': 3,
        'levels_back': 2,
    }
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.randomize(params['levels_back'])

    genome[-2:] = [0, None]  # set inputs for output node to input node
    graph = gp.CGPGraph(genome)

    x = [2.14159]
    y = graph(x)

    assert(abs(x[0] - y[0]) < 1e-15)


# -> primitives
def test_immutable_primitives():
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    with pytest.raises(TypeError):
        primitives[0] = gp.CGPAdd

    with pytest.raises(TypeError):
        primitives._primitives[0] = gp.CGPAdd


# -> primitives
def test_max_arity():
    plain_primitives = [gp.CGPAdd, gp.CGPSub, gp.CGPConstantFloat]
    primitives = gp.CGPPrimitives(plain_primitives)

    arity = 0
    for p in plain_primitives:
        if arity < p._arity:
            arity = p._arity

    assert(arity == primitives.max_arity)


# -> graph
def test_compile_simple():
    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(2, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(x[0] + x[1] - y[0]) < 1e-15)

    primitives = gp.CGPPrimitives([gp.CGPSub])
    genome = gp.CGPGenome(2, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(x[0] - x[1] - y[0]) < 1e-15)


# -> graph
def test_compile_two_columns():
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    genome = gp.CGPGenome(2, 1, 2, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 1, 0, 2, -2, 3, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(x[0] - (x[0] + x[1]) - y[0]) < 1e-15)


# -> graph
def test_compile_two_columns_two_rows():
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    genome = gp.CGPGenome(2, 2, 2, 2, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 1, 0, 1, 0, 0, 2, 0, 2, 3, -2, 4, None, -2, 5, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(x[0] + (x[0] + x[1]) - y[0]) < 1e-15)
    assert(abs((x[0] + x[1]) + (x[0] - x[1]) - y[1]) < 1e-15)


# -> node
def test_compile_addsubmul():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 2,
        'n_rows': 2,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.dna = [
        -1, None, None, -1, None, None,
        2, 0, 1, 1, 0, 1,
        1, 2, 3, 0, 0, 0,
        -2, 4, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(((x[0] * x[1]) - (x[0] - x[1])) - y[0]) < 1e-15)


# -> graph
def test_compile_torch_and_backprop():
    primitives = gp.CGPPrimitives([gp.CGPMul, gp.CGPConstantFloat])
    genome = gp.CGPGenome(1, 1, 2, 2, primitives)
    genome.dna = [-1, None, None, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, -2, 3, None]
    graph = gp.CGPGraph(genome)

    c = graph.compile_torch_class()

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

    assert(loss.detach().numpy() < 1e-15)

    x = [3.]
    x_torch = torch.Tensor(x).view(1, 1)
    assert(abs(c(x_torch)[0].detach().numpy() - graph(x))[0] > 1e-15)
    graph.update_parameter_values(c)
    assert(abs(c(x_torch)[0].detach().numpy() - graph(x))[0] < 1e-15)


def test_cgp():
    params = {
        'seed': 81882,
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 3,
        'n_rows': 3,
        'levels_back': 2,
        'n_mutations': 3,
    }

    np.random.seed(params['seed'])

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPConstantFloat])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.randomize(params['levels_back'])
    graph = gp.CGPGraph(genome)

    history_loss = []
    for i in range(10000):

        genome.mutate(params['n_mutations'], params['levels_back'])
        graph.parse_genome(genome)
        f = graph.compile_func()

        history_loss_trial = []
        for j in range(10):
            x = np.random.randint(1, 10, 2)
            y = f(x)
            loss = (((x[0] * x[1]) - (x[0] - x[1])) - y[0]) ** 2
            history_loss_trial.append(loss)

        if np.sum(history_loss_trial) < 1e-15:
            print(graph[-1].output_str)

        history_loss.append(np.sum(np.mean(history_loss_trial)))

    plt.plot(history_loss)
    plt.show()
