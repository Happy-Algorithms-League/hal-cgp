import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../')
import gp


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


def test_cgp():
    params = {
        'seed': 1234,
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 3,
        'n_rows': 2,
        'levels_back': 2,
    }

    np.random.seed(params['seed'])

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
    genome.randomize(primitives, params['levels_back'])
    graph = gp.CGPGraph(genome, primitives)

    x = [5., 2.]
    y = graph(x)
    print(genome._dna)
    print(x, '->', y)
