"""
Example demonstrating the use of the caching decorator with functional equivalance checking
===========================================================================================

Caches the results of fitness evaluations in a pickle file
(``example_fec_caching_cache.pkl``). To illustrate its practical use,
compare the runtime of this script when you first call it vs. the
second time and when you comment out the decorator on
`inner_objective`.

"""

import functools
import multiprocessing as mp
import time

import numpy as np

import cgp

# %%
# We define the target function for this example.


def f_target(x):
    return x ** 2 + x + 1.0


# %%
# We then define the objective function for the evolutionary
# algorithm: It consists of an inner objective which we wrap with the
# caching decorator. This decorator specifies a pickle file that will
# be used for caching results of fitness evaluations. In addition the
# decorator accepts keyword arguments that specifiy the statistic of
# the samples used for evaluation. The inner objective is then used by
# the objective function to compute (or retrieve from cache) the
# fitness of the individual.


@cgp.utils.disk_cache(
    "example_fec_caching_cache.pkl",
    compute_key=functools.partial(
        cgp.utils.compute_key_from_numpy_evaluation_and_args,
        _seed=12345,
        _min_value=-10.0,
        _max_value=10.0,
        _batch_size=5,
    ),
    file_lock=mp.Lock(),
)
def inner_objective(ind):
    """The caching decorator uses the return values generated from
    providing random inputs to ind.to_numpy() to identify functionally
    indentical individuals and avoid reevaluating them. Note that
    caching only makes sense for deterministic objective functions, as
    it assumes that identical phenotypes will always return the same
    fitness values.

    """
    f = ind.to_func()

    loss = []
    for x_0 in np.linspace(-2.0, 2.0, 100):
        y = f([x_0])
        loss.append((f_target(x_0) - y) ** 2)

    time.sleep(0.25)  # emulate long fitness evaluation

    return np.mean(loss)


def objective(individual):
    if not individual.fitness_is_None():
        return individual

    individual.fitness = -inner_objective(individual)

    return individual


# %%
# Next, we define the parameters for the population, the genome of
# individuals, and the evolutionary algorithm.


params = {
    "population_params": {"n_parents": 10, "seed": 8188211},
    "ea_params": {
        "n_offsprings": 10,
        "tournament_size": 1,
        "mutation_rate": 0.05,
        "n_processes": 1,
    },
    "genome_params": {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 10,
        "n_rows": 2,
        "levels_back": 2,
        "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),
    },
    "evolve_params": {"max_generations": 200, "min_fitness": -1e-12},
}

# %%
# We then create a Population instance and instantiate the evolutionary algorithm.


pop = cgp.Population(**params["population_params"], genome_params=params["genome_params"])
ea = cgp.ea.MuPlusLambda(**params["ea_params"])

# %%
# Finally, we call the `evolve` method to perform the evolutionary search.


cgp.evolve(pop, objective, ea, **params["evolve_params"], print_progress=True)


print(f"evolved function: {pop.champion.to_sympy()}")
