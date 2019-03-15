import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../')
import gp

SEED = np.random.randint(2 ** 31)

params = {
    'n_parents': 5,
    'n_offsprings': 5,
    'genome_length': 10,
    'generations': 100,
    'n_breeding': 5,
    'tournament_size': 6,
    'mutation_rate': 0.1,
}


def test_binary_population():

    np.random.seed(SEED + 123)

    target_sequence = str(np.random.randint(10 ** params['genome_length'])).zfill(params['genome_length'])

    def objective(individual):

        if individual.fitness is not None:
            return individual

        individual.fitness = sum(1 if individual.genome[i] == target_sequence[i] else 0
                                 for i in range(len(target_sequence)))
        return individual

    # create population object that will be evolved
    pop = gp.BinaryPopulation(
        params['n_parents'], params['n_offsprings'], params['n_breeding'],
        params['tournament_size'], params['mutation_rate'], SEED, params['genome_length'])

    # generate initial parent population of size N
    pop.generate_random_parent_population()

    # generate initial offspring population of size N
    pop.generate_random_offspring_population()

    # perform evolution
    for i in range(params['generations']):

        # combine parent and offspring populations
        pop.create_combined_population()

        #  compute fitness for all objectives for all individuals
        pop.compute_fitness(objective)

        # sort population according to fitness & crowding distance
        pop.sort()

        # fill new parent population according to sorting
        pop.create_new_parent_population()

        # generate new offspring population from parent population
        pop.create_new_offspring_population()

        if abs(pop.champion.fitness) < 1e-10:
            break

    assert target_sequence == pop.champion.genome, SEED
