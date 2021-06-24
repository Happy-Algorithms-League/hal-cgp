from typing import Callable, Optional

import numpy as np

from .ea import MuPlusLambda
from .individual import IndividualBase
from .population import Population


def evolve(
    objective: Callable[[IndividualBase], IndividualBase],
    pop: Population = Population(),
    ea: MuPlusLambda = MuPlusLambda(),
    termination_fitness: float = np.inf,
    max_generations: int = np.iinfo(np.int64).max,
    max_objective_calls: int = np.iinfo(np.int64).max,
    print_progress: Optional[bool] = False,
    callback: Optional[Callable[[Population], None]] = None,
) -> Population:
    """
    Evolves a population and returns the history of fitness of parents.

    Parameters
    ----------
    objective : Callable
        An objective function used for the evolution. Needs to take an
        individual (Individual) as input parameter and return
        a modified individual (with updated fitness).
    pop : Population, optional
        A population class that will be evolved. Defaults to population with default parameters.
    ea : EA algorithm instance, optional
        The evolutionary algorithm. Defaults to MuPlusLambda.
    termination_fitness : float, optional
        Minimum fitness at which the evolution is terminated. Defaults to positive infinity.
    max_generations : int, optional
        Maximum number of generations.
        If neither this nor `max_objective_calls` are set, `max_generations` defaults to 1000.
    max_objective_calls: int, optional
        Maximum number of function evaluations.
        Defaults to largest representable integer.
        If neither this nor `max_generations` are set, `max_generations` defaults to 1000.
    print_progress : boolean, optional
        Switch to print out the progress of the algorithm. Defaults to False.
    callback :  callable, optional
        Called after each iteration with the population instance.
        Defaults to None.

    Returns
    -------
    Population
        The evolved population.
    """
    if max_generations == np.iinfo(np.int64).max and max_objective_calls == np.iinfo(np.int64).max:
        max_generations = 1000

    ea.initialize_fitness_parents(pop, objective)
    if callback is not None:
        callback(pop)

    # perform evolution
    max_fitness = np.finfo(float).min
    # Main loop: -1 offset since the last loop iteration will still increase generation by one
    while pop.generation < max_generations - 1 and ea.n_objective_calls < max_objective_calls:

        pop = ea.step(pop, objective)

        # progress printing, recording, checking exit condition etc.; needs to
        # be done /after/ new parent population was populated from combined
        # population and /before/ new individuals are created as offsprings
        assert isinstance(pop.champion.fitness, float)
        if pop.champion.fitness > max_fitness:
            max_fitness = pop.champion.fitness

        if print_progress:
            if max_generations < np.iinfo(np.int64).max:
                print(
                    f"\r[{pop.generation + 1}/{max_generations}] max fitness: {max_fitness}\033[K",
                    end="",
                    flush=True,
                )
            elif max_objective_calls < np.iinfo(np.int64).max:
                print(
                    f"\r[{ea.n_objective_calls}/{max_objective_calls}] "
                    f"max fitness: {max_fitness}\033[K",
                    end="",
                    flush=True,
                )
            else:
                assert False  # should never be reached

        if callback is not None:
            callback(pop)

        if pop.champion.fitness + 1e-10 >= termination_fitness:
            break

    if print_progress:
        print()

    return pop
