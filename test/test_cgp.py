import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../')
import gp


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

    primitives = gp.CGPPrimitives()
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives, params['levels_back'])
    graph = gp.CGPGraph(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'])
    graph.parse_genome(genome, primitives)

    y = graph([2., 5.])
    print(y)
