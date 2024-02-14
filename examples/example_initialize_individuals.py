"""
Minimal example for fixed initial conditions
============================================

Example demonstrating the use of Cartesian genetic programming for a simple
regression task (see `example_minimal.py`). However, here we initialize the
initial parent population to a specific expression.
"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
   Usage:
     example_initialize_individuals.py

   Options:
     -h --help
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
from docopt import docopt

import cgp

args = docopt(docopt_str)

# %%
# We first define a target function.


def f_target(x):
    return x ** 2 + 1.0


# %%
# Then we define the objective function for the evolution. It uses
# the mean-squared error between the output of the expression
# represented by a given individual and the target function evaluated
# on a set of random points.


def objective(individual):

    if not individual.fitness_is_None():
        return individual

    n_function_evaluations = 1000

    np.random.seed(1234)

    f = individual.to_func()
    loss = 0
    for x in np.random.uniform(-4, 4, n_function_evaluations):
        # the callable returned from `to_func` accepts and returns
        # lists; accordingly we need to pack the argument and unpack
        # the return value
        y = f(x)
        loss += (f_target(x) - y) ** 2

    individual.fitness = -loss / n_function_evaluations

    return individual


# %%
# We want to initialize all individuals to the same expression (for
# illustration, functionally not necessarily the best idea). We can provide a
# function to the `Population` constructor which is called for each individual
# of the initial parent population. Unfortunately, we can not provide the
# expression directly, but rather need to manually set (parts of) the
# individuals genome to the correct values. See Jordan, Schmidt et al. (2020)
# https://doi.org/10.7554/eLife.66273 figure 2 for details about the encoding.
def individual_init(ind):
    # f(x) = x * x
    ind.genome.set_expression_for_output([2, 0, 0])
    assert str(ind.to_sympy(simplify=False)) == "x_0*x_0"
    return ind


pop = cgp.Population(individual_init=individual_init)

# %%
# Next, we set up the evolutionary search. We define a callback for recording
# of fitness over generations
history = {}
history["fitness_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)


# %%
# and finally perform the evolution relying on the libraries default
# hyperparameters except that we terminate the evolution as soon as one
# individual has reached fitness zero.
pop = cgp.evolve(
    objective, pop, termination_fitness=0.0, print_progress=True, callback=recording_callback
)


# %%
# After finishing the evolution, we plot the result and log the final
# evolved expression.


width = 9.0
fig, axes = plt.subplots(1, 2, figsize=(width, width / scipy.constants.golden))

ax_fitness, ax_function = axes[0], axes[1]
ax_fitness.set_xlabel("Generation")
ax_fitness.set_ylabel("Fitness")

ax_fitness.plot(history["fitness_champion"], label="Champion")

ax_fitness.set_yscale("symlog")
ax_fitness.set_ylim(-1.0e2, 0.1)
ax_fitness.axhline(0.0, color="0.7")

f = pop.champion.to_func()
x = np.linspace(-5.0, 5.0, 20)
y = [f(x_i) for x_i in x]
y_target = [f_target(x_i) for x_i in x]

ax_function.plot(x, y_target, lw=2, alpha=0.5, label="Target")
ax_function.plot(x, y, "x", label="Champion")
ax_function.legend()
ax_function.set_ylabel(r"$f(x)$")
ax_function.set_xlabel(r"$x$")

fig.savefig("example_initialize_individuals.pdf", dpi=300)
