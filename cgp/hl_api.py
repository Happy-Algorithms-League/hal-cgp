import numpy as np
from typing import Optional, Callable

from .population import Population
from .individual import IndividualBase
from .ea import MuPlusLambda


def evolve(
    pop: Population,
    objective: Callable[[IndividualBase], IndividualBase],
    ea: MuPlusLambda,
    max_generations: int,
    min_fitness: float,
    print_progress: Optional[bool] = False,
    callback: Optional[Callable[[Population], None]] = None,
    n_processes: int = 1,
) -> None:
    """
    Evolves a population and returns the history of fitness of parents.

    Parameters
    ----------
    pop : Population
        A population class that will be evolved.
    objective : Callable
        An objective function used for the evolution. Needs to take an
        individual (Individual) as input parameter and return
        a modified individual (with updated fitness).
    ea : EA algorithm instance
        The evolution algorithm. Needs to be a class instance with an
        `initialize_fitness_parents` and `step` method.
    max_generations : int
        Maximum number of generations.
    min_fitness : float
        Minimum fitness at which the evolution is stopped.
    print_progress : boolean, optional
        Switch to print out the progress of the algorithm. Defaults to False.
    callback :  callable, optional
        Called after each iteration with the population instance.
        Defaults to None.
    n_processes : int, optional
        Number of parallel processes to be used. If greater than 1,
        parallel evaluation of the objective is supported. Currently
        not implemented. Defaults to 1.

    Returns
    -------
    None
    """

    ea.initialize_fitness_parents(pop, objective)
    if callback is not None:
        callback(pop)

    # perform evolution
    max_fitness = np.finfo(np.float).min
    # Main loop: -1 offset since the last loop iteration will still increase generation by one
    while pop.generation < max_generations - 1:

        pop = ea.step(pop, objective)

        # progress printing, recording, checking exit condition etc.; needs to
        # be done /after/ new parent population was populated from combined
        # population and /before/ new individuals are created as offsprings
        assert isinstance(pop.champion.fitness, float)
        if pop.champion.fitness > max_fitness:
            max_fitness = pop.champion.fitness

        if print_progress:
            print(
                f"\r[{pop.generation + 1}/{max_generations}] max fitness: {max_fitness}\033[K",
                end="",
                flush=True,
            )

        if callback is not None:
            callback(pop)

        if pop.champion.fitness + 1e-10 >= min_fitness:
            break

    if print_progress:
        print()
