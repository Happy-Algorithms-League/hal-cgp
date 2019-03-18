import numpy as np
import pytest
import sys

sys.path.insert(0, '../')
import gp


SEED = np.random.randint(2 ** 31)


def test_add():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = [gp.CGPAdd]
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert abs(x[0] + x[1] - y[0]) < 1e-15


def test_sub():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = [gp.CGPSub]
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert abs(x[0] - x[1] - y[0]) < 1e-15


def test_mul():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = [gp.CGPMul]
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert abs((x[0] * x[1]) - y[0]) < 1e-15
