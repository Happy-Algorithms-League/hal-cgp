"""
Example for evolutionary regression with local search via evolution strategies
==============================================================================

Example demonstrating the use of Cartesian genetic programming for a
regression task that involves numeric constants. Local search via
evolution strategies is used to determine numeric leaf values of the
graph.
"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
  Usage:
    example_local_search_evolution_strategies.py [--max-generations=<N>]

  Options:
    -h --help
    --max-generations=<N>  Maximum number of generations [default: 500]

"""

import functools

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from docopt import docopt

import cgp

args = docopt(docopt_str)

# %%
# We first define the target function. Note that this function contains
# numeric values which are initially not available as constants to the search.


def f_target(x):
    return np.e * x[:, 0] ** 2 + 1.0 + np.pi


# %%
# Then we define the objective function for the evolution. It consists
# of an inner objective which accepts a NumPy-compatible function as
# its first argument and returns the mean-squared error between the
# expression represented by a given individual and the target function
# evaluated on a set of random points. This inner objective is used by
# the local search to determine appropriate values for Parameter node
# and the actual objective function to update the fitness of the
# individual.


def inner_objective(ind, seed):
    """Return a loss for the numpy-compatible function f. Used for
    calculating the fitness of each individual and for the local
    search of numeric leaf values.

    """

    f = ind.to_numpy()
    rng = np.random.RandomState(seed)
    batch_size = 500
    x = rng.uniform(-5, 5, size=(batch_size, 1))
    y = f(x)
    return -np.mean((f_target(x) - y[:, 0]) ** 2)


def objective(individual, seed):
    """Objective function of the regression task."""

    if not individual.fitness_is_None():
        return individual

    individual.fitness = inner_objective(individual, seed)

    return individual


# %%
# Next, we define the parameters for the population, the genome of
# individuals, the evolutionary algorithm, and the local search.


population_params = {"n_parents": 1, "seed": 818821}

genome_params = {
    "n_inputs": 1,
    "n_outputs": 1,
    "n_columns": 36,
    "n_rows": 1,
    "levels_back": None,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.Parameter),
}

ea_params = {
    "n_offsprings": 4,
    "mutation_rate": 0.05,
    "tournament_size": 1,
    "n_processes": 1,
    "k_local_search": 2,
}

evolve_params = {"max_generations": int(args["--max-generations"]), "min_fitness": 0.0}

# restrict the number of steps in the local search; since parameter
# values are propagated from parents to offsprings, parameter values
# may be iteratively improved across generations despite the small
# number of steps per generation
local_search_params = {"max_steps": 5}

# %%
# We then create a Population instance and instantiate the local search
# and evolutionary algorithm.

pop = cgp.Population(**population_params, genome_params=genome_params)

# define the function for local search; an instance of the
# EvolutionStrategies can be called with an individual as an argument
# for which the local search is performed
local_search = cgp.local_search.EvolutionStrategies(
    objective=functools.partial(inner_objective, seed=population_params["seed"] + 1),
    seed=population_params["seed"] + 2,
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


obj = functools.partial(objective, seed=population_params["seed"] + 1)

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


print(f"Final expression {pop.champion.to_sympy()[0]} with fitness {pop.champion.fitness}")

history_fitness = np.array(history["fitness_parents"])
ax_fitness.plot(np.max(history_fitness, axis=1), label="Champion")
ax_fitness.plot(np.mean(history_fitness, axis=1), label="Population mean")

x = np.linspace(-5.0, 5, 100).reshape(-1, 1)
f = pop.champion.to_func()
y = [f(xi) for xi in x]
ax_function.plot(x, f_target(x), lw=2, label="Target")
ax_function.plot(x, y, lw=1, label="Target", marker="x")

plt.savefig("example_local_search_evolution_strategies.pdf", dpi=300)
