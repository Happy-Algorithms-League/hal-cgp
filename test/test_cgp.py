import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys

sys.path.insert(0, '../')
import gp


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
    genome.dna = [-1, 0, 0, -1, 0, 0, 0, -2, -1, -2, 0, 0]

    for region in genome.input_regions():
        assert(region == [-1, 0, 0])

    for region in genome.hidden_regions():
        assert(region == [0, -2, -1])

    for region in genome.output_regions():
        assert(region == [-2, 0, 0])


def test_add():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.dna = [0, -2, -1, 0]
    graph = gp.CGPGraph(genome, primitives)

    x = [5., 1.5]
    y = graph(x)

    assert(abs(x[0] + x[1] - y[0]) < 1e-15)


def test_sub():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPSub])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.dna = [0, -2, -1, 0]
    graph = gp.CGPGraph(genome, primitives)

    x = [5., 1.5]
    y = graph(x)

    assert(abs(x[0] - x[1] - y[0]) < 1e-15)


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
    genome._dna[-1] = -1
    graph = gp.CGPGraph(genome, primitives)

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

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPConstantFloat])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.randomize(params['levels_back'])
    graph = gp.CGPGraph(genome, primitives)

    x = [5., 2.]

    history_loss = []
    for i in range(1000):
        genome.mutate(params['n_mutations'], params['levels_back'])
        graph.parse_genome(genome)
        y = graph(x)
        loss = (x[0] + x[1] - y[0]) ** 2
        history_loss.append(loss)

    plt.plot(history_loss)
    plt.show()
