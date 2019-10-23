import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys

sys.path.insert(0, '../')
import gp

SEED = np.random.randint(2 ** 31)


def test_mutation_rate_within_bounds():
    mutation_rate = 0.
    with pytest.raises(ValueError):
        pop = gp.BinaryPopulation(1, mutation_rate, SEED, {'genome_length': 2, 'primitives': [0]})

    mutation_rate = 1.
    with pytest.raises(ValueError):
        pop = gp.BinaryPopulation(1, mutation_rate, SEED, {'genome_length': 2, 'primitives': [0]})

    mutation_rate = 0.5
    pop = gp.BinaryPopulation(1, mutation_rate, SEED, {'genome_length': 2, 'primitives': [0]})


def test_binary_population():

    params = {
        'n_parents': 5,
        'n_offsprings': 5,
        'max_generations': 500,
        'min_fitness': 0.,
        'n_breeding': 5,
        'tournament_size': 6,
        'mutation_rate': 0.1,
        'max_fitness': -1e-8,
    }

    genome_params = {
        'genome_length': 10,
        'primitives': list(range(10)),
    }

    np.random.seed(SEED + 123)

    target_sequence = list(np.random.choice(
        genome_params['primitives'],
        size=genome_params['genome_length']))

    def objective(individual):

        if individual.fitness is not None:
            return individual

        individual.fitness = sum(0 if individual.genome[i] == target_sequence[i] else -1
                                 for i in range(len(target_sequence)))
        return individual

    # create population object that will be evolved
    pop = gp.BinaryPopulation(params['n_parents'], params['mutation_rate'], SEED, genome_params)
    ea = gp.ea.MuPlusLambda(params['n_parents'], params['n_offsprings'], params['n_breeding'], params['tournament_size'])

    gp.evolve(pop, objective, ea, params['max_generations'], params['min_fitness'])

    assert target_sequence == pop.champion.genome.dna, SEED
