"""
Example for differential evolutionary regression
================================================

Example demonstrating the use of Cartesian genetic programming for
a regression task that involves numeric constants. Local
gradient-based search is used to determine numeric leaf values of the
graph.

References:

- Topchy, A., & Punch, W. F. (2001). Faster genetic programming based
  on local gradient search of numeric leaf values. In Proceedings of
  the genetic and evolutionary computation conference (GECCO-2001)
  (Vol. 155162). Morgan Kaufmann San Francisco, CA, USA.

- Izzo, D., Biscani, F., & Mereta, A. (2017). Differentiable genetic
  programming. In European Conference on Genetic Programming
  (pp. 35-51). Springer, Cham.

"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
  Usage:
    example_differential_evo_regression.py [--max-generations=<N>]

  Options:
    -h --help
    --max-generations=<N>  Maximum number of generations [default: 2000]

"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import torch
from docopt import docopt

import cgp

args = docopt(docopt_str)

# %%
# We first define the target function. Note that this function contains
# numeric values which are initially not available as constants to the search.


def f_target(x):
    return x[:, 0] ** 2 + 1.0 + np.pi


# %%
# Then we define the differentiable(!) objective function for the evolution.  It
# consists of an inner objective which accepts a torch tensor as input
# variable and uses mean-squared error between the expression
# represented by a given individual and the target function evaluated
# on a set of random points. This inner objective is then used by
# actual objective function to update the fitness of the individual.


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

    if not individual.fitness_is_None():
        return individual

    f = individual.to_torch()
    loss = inner_objective(f, seed)
    individual.fitness = -loss.item()

    return individual


# %%
# Next, we define the parameters for the population, the genome of
# individuals, the evolutionary algorithm, and the local search.


population_params = {"n_parents": 1, "seed": 818821}

genome_params = {
    "n_inputs": 1,
    "n_outputs": 1,
    "n_columns": 20,
    "n_rows": 1,
    "levels_back": None,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.Parameter),
}

ea_params = {
    "n_offsprings": 4,
    "tournament_size": 1,
    "mutation_rate": 0.05,
    "n_processes": 1,
    "k_local_search": 2,
}

evolve_params = {"max_generations": int(args["--max-generations"]), "termination_fitness": 0.0}

# use an uneven number of gradient steps so they can not easily
# average out for clipped values
local_search_params = {"lr": 1e-3, "gradient_steps": 9}

# %%
# We then create a Population instance and instantiate the local search
# and evolutionary algorithm.

pop = cgp.Population(**population_params, genome_params=genome_params)

# define the function for local search; parameters such as the
# learning rate and number of gradient steps are fixed via the use
# of `partial`; the `local_search` function should only receive a
# population of individuals as input
local_search = functools.partial(
    cgp.local_search.gradient_based,
    objective=functools.partial(inner_objective, seed=population_params["seed"]),
    **local_search_params,
)

ea = cgp.ea.MuPlusLambda(**ea_params, local_search=local_search)

# %%
#  We define a recording callback closure for bookkeeping of the progression of the evolution.


history = {}
history["champion"] = []
history["fitness_parents"] = []


def recording_callback(pop):
    history["champion"].append(pop.champion)
    history["fitness_parents"].append(pop.fitness_parents())


obj = functools.partial(objective, seed=population_params["seed"])

# %%
# Finally, we call the `evolve` method to perform the evolutionary search.

cgp.evolve(pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback)


# %%
# After finishing the evolution, we plot the result and log the final
# evolved expression.

width = 9.0
fig = plt.figure(figsize=(width, width / scipy.constants.golden))

ax_fitness = fig.add_subplot(121)
ax_fitness.set_xlabel("Generation")
ax_fitness.set_ylabel("Fitness")
ax_fitness.set_yscale("symlog")

ax_function = fig.add_subplot(122)
ax_function.set_ylabel(r"$f(x)$")
ax_function.set_xlabel(r"$x$")


print(f"Final expression {pop.champion.to_sympy()} with fitness {pop.champion.fitness}")

history_fitness = np.array(history["fitness_parents"])
ax_fitness.plot(np.max(history_fitness, axis=1), label="Champion")
ax_fitness.plot(np.mean(history_fitness, axis=1), label="Population mean")

x = np.linspace(-5.0, 5, 100).reshape(-1, 1)
f = pop.champion.to_func()
y = [f(xi) for xi in x]
ax_function.plot(x, f_target(x), lw=2, label="Target")
ax_function.plot(x, y, lw=1, label="Target", marker="x")

plt.savefig("example_differential_evo_regression.pdf", dpi=300)
