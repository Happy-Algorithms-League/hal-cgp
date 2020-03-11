import numpy as np


def evolve(pop, objective, ea, max_generations, min_fitness,
           record_history=None, print_progress=False, *, label=None, n_processes=1):
    """
    Evolves a population and returns the history of fitness of parents.
    """

    # data structure for recording evolution history; can be populated via user
    # defined recording function
    history = {}

    ea.initialize_fitness_parents(pop, objective, label=label)
    if record_history is not None:
        record_history(pop, history)

    # perform evolution
    max_fitness = np.finfo(np.float).min
    while pop.generation < max_generations - 1:  # -1 offset since the last loop iteration will still increase generation by one

        pop = ea.step(pop, objective, label=label)

        # progress printing, recording, checking exit condition etc.; needs to
        # be done /after/ new parent population was populated from combined
        # population and /before/ new individuals are created as offsprings
        if pop.champion.fitness > max_fitness:
            max_fitness = pop.champion.fitness

        if print_progress:
            print(f'\r[{pop.generation + 1}/{max_generations}'
                  f'({pop.champion.idx})] max fitness: {max_fitness}\033[K', end='')

        if record_history is not None:
            record_history(pop, history)

        if pop.champion.fitness + 1e-10 >= min_fitness:
            for key in history:
                history[key] = history[key][:pop.generation + 1]
            break

    if print_progress:
        print()

    return history
