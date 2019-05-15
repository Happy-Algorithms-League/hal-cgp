import numpy as np


def evolve(pop, objective, max_generations, min_fitness,
           record_history=None, print_progress=False, *, label=None):
    """
    Evolves a population and returns the history of fitness of parents.
    """

    # data structure for recording evolution history; can be populated via user
    # defined recording function
    history = {}

    # generate initial parent population of size N
    pop.generate_random_parent_population()

    # generate initial offspring population of size N
    pop.generate_random_offspring_population()

    # perform evolution
    max_fitness = -1e15
    for generation in range(max_generations):

        # combine parent and offspring populations
        pop.create_combined_population()

        #  compute fitness for all objectives for all individuals
        pop.compute_fitness(objective, label=label)

        # sort population according to fitness & crowding distance
        pop.sort()

        # fill new parent population according to sorting
        pop.create_new_parent_population()

        # progress printing, recording, checking exit condition etc.; needs to
        # be done /after/ new parent population was populated from combined
        # population and /before/ new individuals are created as offsprings
        if pop.champion.fitness > max_fitness:
            max_fitness = pop.champion.fitness

        if print_progress:
            print(f'\r[{generation + 1}/{max_generations}'
                  f'({pop.champion.idx})] max fitness: {max_fitness}\033[K', end='')

        if record_history is not None:
            record_history(pop, history)

        if pop.champion.fitness + 1e-10 >= min_fitness:
            for key in history:
                history[key] = history[key][:generation + 1]
            break

        # generate new offspring population from parent population
        pop.create_new_offspring_population()

    if print_progress:
        print()

    return history
