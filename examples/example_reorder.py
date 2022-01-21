"""
Example for evolutionary regression with genome reordering
===========================================================

Example demonstrating the effect of genome reordering.

References
-----------
Goldman B.W., Punch W.F. (2014): Analysis of Cartesian Genetic Programming’s
Evolutionary Mechanisms
DOI: 10.1109/TEVC.2014.2324539
"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
   Usage:
     example_reorder.py [--max-generations=<N>]

   Options:
     -h --help
     --max-generations=<N>  Maximum number of generations [default: 300]
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

    n_function_evaluations = 10000

    np.random.seed(12345)

    f = individual.to_numpy()
    x = np.random.uniform(-4, 4, n_function_evaluations)
    y = f(x)
    loss = np.mean((f_target(x) - y) ** 2)
    individual.fitness = -loss

    return individual


# %%
# Next, we set up the evolutionary search. We first define the parameters for
# the genome of individuals, and two evolutionary algorithms without (default)
# and with genome reordering.

seed = 1234

ea_params_with_reorder = {"reorder_genome": True}

evolve_params = {"max_generations": int(args["--max-generations"]), "termination_fitness": 0.0}

# %%
# We create two populations that will be evolved
pop = cgp.Population()
pop_with_reorder = cgp.Population()

# %%
# and an instance of the (mu + lambda) evolutionary algorithm
ea_with_reorder = cgp.ea.MuPlusLambda(**ea_params_with_reorder)

# %%
# We define two callbacks for recording of fitness over generations
history = {}
history["fitness_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)


history_with_reorder = {}
history_with_reorder["fitness_champion"] = []


def recording_callback_with_reorder(pop):
    history_with_reorder["fitness_champion"].append(pop.champion.fitness)


# %%
# and finally perform the evolution of the two populations
pop = cgp.evolve(
    objective, pop, **evolve_params, print_progress=True, callback=recording_callback,
)

pop_with_reorder = cgp.evolve(
    objective,
    pop_with_reorder,
    ea_with_reorder,
    **evolve_params,
    print_progress=True,
    callback=recording_callback_with_reorder,
)


# %%
# After finishing the evolution, we plot the evolution of the fittest individual
# with and without genome reordering


width = 9.0
fig = plt.figure(1, figsize=[width, width / scipy.constants.golden])

plt.plot(history["fitness_champion"], label="Champion")
plt.plot(history_with_reorder["fitness_champion"], label="Champion with reorder")

plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.legend(["Champion", "Champion with reorder"])

plt.yscale("symlog")
plt.ylim(-1.0e2, 0.1)
plt.axhline(0.0, color="0.7")

fig.savefig("example_reorder.pdf", dpi=300)
