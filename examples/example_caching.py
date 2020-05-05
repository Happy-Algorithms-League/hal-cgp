import numpy as np
import time

import gp

""" Example demonstrating the use of the caching decorator.

Caches the results of fitness evaluations in a pickle file
('example_caching_cache.pkl'). To illustrate its practical use,
compare the runtime of this script when you first call it vs. the
second time and when you comment out the decorator on
`inner_objective`."""


def f_target(x):
    return x ** 2 + x + 1.0


@gp.utils.disk_cache("example_caching_cache.pkl")
def inner_objective(expr):
    """The caching decorator uses the function parameters to identify
    identical function calls. Here, as many different genotypes
    produce the same simplified SymPy expression we can use such
    expressions as an argument to the decorated function to avoid
    reevaluating functionally identical individuals.
    Note that caching only makes sense for deterministic objective
    functions, as it assumes that identical expressions will always
    return the same fitness values.

    """
    loss = []
    for x0 in np.linspace(-2.0, 2.0, 100):
        y = float(expr[0].subs({"x_0": x0}).evalf())
        loss.append((f_target(x0) - y) ** 2)

    time.sleep(0.25)  # emulate long fitness evaluation

    return np.mean(loss)


def objective(individual):
    if individual.fitness is not None:
        return individual

    individual.fitness = -inner_objective(individual.to_sympy())

    return individual


def evolution():
    params = {
        "population_params": {"n_parents": 10, "mutation_rate": 0.5, "seed": 8188211},
        "ea_params": {
            "n_offsprings": 10,
            "n_breeding": 10,
            "tournament_size": 1,
            "n_processes": 1,
        },
        "genome_params": {
            "n_inputs": 1,
            "n_outputs": 1,
            "n_columns": 10,
            "n_rows": 2,
            "levels_back": 2,
            "primitives": [gp.Add, gp.Sub, gp.Mul, gp.ConstantFloat],
        },
        "evolve_params": {"max_generations": 100, "min_fitness": -1e-12},
    }

    pop = gp.Population(**params["population_params"], genome_params=params["genome_params"])
    ea = gp.ea.MuPlusLambda(**params["ea_params"])

    gp.evolve(pop, objective, ea, **params["evolve_params"], print_progress=True)

    return pop.champion


if __name__ == "__main__":
    champion = evolution()
    print(f"evolved function: {champion.to_sympy()}")
