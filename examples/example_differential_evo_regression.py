import functools
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import torch

import cgp


"""Example demonstrating the use of Cartesian Genetic Programming for
a regression task that involves numeric constants. Local
gradient-based search is used to determine numeric leaf values of the
graph.

References:

Topchy, A., & Punch, W. F. (2001). Faster genetic programming
based on local gradient search of numeric leaf values. In Proceedings
of the genetic and evolutionary computation conference (GECCO-2001)
(Vol. 155162). Morgan Kaufmann San Francisco, CA, USA.

Izzo, D., Biscani, F., & Mereta, A. (2017). Differentiable
genetic programming. In European Conference on Genetic Programming
(pp. 35-51). Springer, Cham.

"""


def f_target(x):
    return x[:, 0] ** 2 + 1.0 + np.pi


def inner_objective(f, seed):
    """Return a differentiable loss of the differentiable graph f. Used
    for calculating the fitness of each individual and for the local
    search of numeric leaf values.

    """

    torch.manual_seed(seed)
    batch_size = 500
    x = torch.DoubleTensor(batch_size, 1).uniform_(-5, 5)
    y = f(x)
    return torch.nn.MSELoss()(f_target(x), y[:, 0])


def objective(individual, seed):
    """Objective function of the regression task."""

    if individual.fitness is not None:
        return individual

    f = individual.to_torch()
    loss = inner_objective(f, seed)
    individual.fitness = -loss.item()

    return individual


def evolution():

    population_params = {"n_parents": 1, "mutation_rate": 0.05, "seed": 818821}

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 20,
        "n_rows": 1,
        "levels_back": None,
        "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.Parameter),
    }

    ea_params = {"n_offsprings": 4, "n_breeding": 4, "tournament_size": 1, "n_processes": 1}

    evolve_params = {"max_generations": 2000, "min_fitness": 0.0}

    # use an uneven number of gradient steps so they can not easily
    # average out for clipped values
    local_search_params = {"lr": 1e-3, "gradient_steps": 9}

    pop = cgp.Population(**population_params, genome_params=genome_params)

    # define the function for local search; parameters such as the
    # learning rate and number of gradient steps are fixed via the use
    # of `partial`; the local_search function should only receive a
    # population of individuals as input
    local_search = functools.partial(
        cgp.local_search.gradient_based,
        objective=functools.partial(inner_objective, seed=population_params["seed"]),
        **local_search_params,
    )

    ea = cgp.ea.MuPlusLambda(**ea_params, local_search=local_search)

    history = {}
    history["champion"] = []
    history["fitness_parents"] = []

    def recording_callback(pop):
        history["champion"].append(pop.champion)
        history["fitness_parents"].append(pop.fitness_parents())

    obj = functools.partial(objective, seed=population_params["seed"])

    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback,
    )

    return history, pop.champion


if __name__ == "__main__":

    width = 9.0

    fig = plt.figure(figsize=(width, width / scipy.constants.golden))

    ax_fitness = fig.add_subplot(121)
    ax_fitness.set_xlabel("Generation")
    ax_fitness.set_ylabel("Fitness")
    ax_fitness.set_yscale("symlog")

    ax_function = fig.add_subplot(122)
    ax_function.set_ylabel(r"$f(x)$")
    ax_function.set_xlabel(r"$x$")

    history, champion = evolution()

    print(f"Final expression {champion.to_sympy()[0]} with fitness {champion.fitness}")

    history_fitness = np.array(history["fitness_parents"])
    ax_fitness.plot(np.max(history_fitness, axis=1), label="Champion")
    ax_fitness.plot(np.mean(history_fitness, axis=1), label="Population mean")

    x = np.linspace(-5.0, 5, 100).reshape(-1, 1)
    f = champion.to_func()
    y = [f(xi) for xi in x]
    ax_function.plot(x, f_target(x), lw=2, label="Target")
    ax_function.plot(x, y, lw=1, label="Target", marker="x")

    plt.savefig("example_differential_evo_regression.pdf", dpi=300)
