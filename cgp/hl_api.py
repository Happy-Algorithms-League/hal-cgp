from typing import Callable, Optional

import numpy as np

from .ea import MuPlusLambda
from .individual import IndividualBase
from .population import Population


def evolve(
    pop: Population,
    objective: Callable[[IndividualBase], IndividualBase],
    ea: MuPlusLambda,
    min_fitness: float,
    max_generations: int = np.inf,
    max_objective_calls: int = np.inf,
    print_progress: Optional[bool] = False,
    callback: Optional[Callable[[Population], None]] = None,
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
    min_fitness : float
        Minimum fitness at which the evolution is stopped.
    max_generations : int
        Maximum number of generations.
        Defaults to positive infinity.
        Either this or `max_objective_calls` needs to be set to a finite value.
    max_objective_calls: int
        Maximum number of function evaluations.
        Defaults to positive infinity.
    print_progress : boolean, optional
        Switch to print out the progress of the algorithm. Defaults to False.
    callback :  callable, optional
        Called after each iteration with the population instance.
        Defaults to None.

    Returns
    -------
    None
    """
    if np.isinf(max_generations) and np.isinf(max_objective_calls):
        raise ValueError("Either max_generations or max_objective_calls must be finite.")

    ea.initialize_fitness_parents(pop, objective)
    if callback is not None:
        callback(pop)

    # perform evolution
    max_fitness = np.finfo(np.float).min
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
            if np.isfinite(max_generations):
                print(
                    f"\r[{pop.generation + 1}/{max_generations}] max fitness: {max_fitness}\033[K",
                    end="",
                    flush=True,
                )
            elif np.isfinite(max_objective_calls):
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

        if pop.champion.fitness + 1e-10 >= min_fitness:
            break

    if print_progress:
        print()
