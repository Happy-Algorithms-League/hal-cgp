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

    primitives = [gp.CGPAdd]
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

    primitives = [gp.CGPAdd]
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

    primitives = [gp.CGPAdd]
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 0, None]

    for region_idx, region in genome.iter_input_regions():
        assert(region == [-1, None, None])

    for region_idx, region in genome.iter_hidden_regions():
        assert(region == [0, 0, 1])

    for region_idx, region in genome.iter_output_regions():
        assert(region == [-2, 0, None])


def test_check_levels_back_consistency():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 4,
        'n_rows': 3,
        'levels_back': None,
    }

    primitives = [gp.CGPAdd]

    params['levels_back'] = 0
    with pytest.raises(ValueError):
        gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)

    params['levels_back'] = params['n_columns'] + 1
    with pytest.raises(ValueError):
        gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)

    params['levels_back'] = params['n_columns'] - 1
    gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)


def test_catch_no_non_coding_allele_in_non_coding_region():
    primitives = [gp.CGPConstantFloat]
    genome = gp.CGPGenome(1, 1, 1, 1, 1, primitives)

    # should raise error: ConstantFloat node has no inputs, but input gene has
    # value different from the non-coding allele
    with pytest.raises(ValueError):
        genome.dna = [-1, None, 0, 0, -2, 1]

    # correct
    genome.dna = [-1, None, 0, None, -2, 1]


def test_individuals_have_different_genomes():

    population_params = {
        'n_parents': 5,
        'n_offspring': 5,
        'generations': 50000,
        'n_breeding': 5,
        'tournament_size': 2,
        'mutation_rate': 0.05,
    }

    genome_params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 6,
        'n_rows': 6,
        'levels_back': 2,
        'primitives': [gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPDiv, gp.CGPConstantFloat],
    }

    pop = gp.CGPPopulation(
        population_params['n_parents'], population_params['n_offspring'], population_params['n_breeding'], population_params['tournament_size'], population_params['mutation_rate'], SEED, genome_params)

    pop.generate_random_parent_population()
    pop.generate_random_offspring_population()

    for i, ind in enumerate(pop):
        ind.fitness = -i

    pop.create_combined_population()

    for i, ind in enumerate(pop._combined):
        ind.fitness = -i

    pop.sort()

    pop.create_new_parent_population()
    pop.create_new_offspring_population()

    for i, parent_i in enumerate(pop._parents):

        for j, parent_j in enumerate(pop._parents):
            if i != j:
                assert parent_i is not parent_j
                assert parent_i.genome is not parent_j.genome
                assert parent_i.genome.dna is not parent_j.genome.dna

        for j, offspring_j in enumerate(pop._offsprings):
            if i != j:
                assert parent_i is not offspring_j
                assert parent_i.genome is not offspring_j.genome
                assert parent_i.genome.dna is not offspring_j.genome.dna
