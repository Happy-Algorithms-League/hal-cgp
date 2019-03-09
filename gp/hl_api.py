import numpy as np


def evolve(pop, objective, max_generations, min_fitness, *, label=None):
    """
    Evolves a population and returns the history of fitness of parents.
    """

    # generate initial parent population of size N
    pop.generate_random_parent_population()

    # generate initial offspring population of size N
    pop.generate_random_offspring_population()

    # perform evolution
    history_fitness = np.empty((max_generations, pop._n_parents))
    history_dna_parents = np.empty((max_generations, pop._n_parents, len(pop._parents[0].genome.dna)))
    for generation in range(max_generations):

        # combine parent and offspring populations
        pop.create_combined_population()

        #  compute fitness for all objectives for all individuals
        pop.compute_fitness(objective, label=label)

        # sort population according to fitness & crowding distance
        pop.sort()

        # fill new parent population according to sorting
        pop.create_new_parent_population()

        # generate new offspring population from parent population
        pop.create_new_offspring_population()

        # perform local search to tune values of constants
        # TODO pop.local_search(objective)

        history_fitness[generation] = pop.fitness_parents()
        history_dna_parents[generation] = pop.dna_parents()

        if np.mean(pop.fitness_parents()) + 1e-10 >= min_fitness:
            break

    return history_fitness[:generation + 1], history_dna_parents[:generation + 1]
