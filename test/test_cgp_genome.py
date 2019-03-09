import numpy as np
import pytest
import sys

sys.path.insert(0, '../')
import gp


SEED = np.random.randint(2 ** 31)


def test_check_dna_consistency():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
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


def test_permissable_inputs():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 4,
        'n_rows': 3,
        'levels_back': 2,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.randomize(np.random)

    for input_idx in range(params['n_inputs']):
        region_idx = input_idx
        with pytest.raises(AssertionError):
            genome._permissable_inputs(region_idx)

    expected_for_hidden = [
        [0, 1],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 5, 6, 7, 8, 9, 10],
    ]

    for column_idx in range(params['n_columns']):
        region_idx = params['n_inputs'] + params['n_rows'] * column_idx
        assert expected_for_hidden[column_idx] == genome._permissable_inputs(region_idx)

    expected_for_output = list(range(params['n_inputs'] + params['n_rows'] * params['n_columns']))

    for output_idx in range(params['n_outputs']):
        region_idx = params['n_inputs'] + params['n_rows'] * params['n_columns'] + output_idx
        assert expected_for_output == genome._permissable_inputs(region_idx)


def test_region_iterators():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 0, None]

    for region_idx, region in genome.iter_input_regions():
        assert(region == [-1, None, None])

    for region_idx, region in genome.iter_hidden_regions():
        assert(region == [0, 0, 1])

    for region_idx, region in genome.iter_output_regions():
        assert(region == [-2, 0, None])


