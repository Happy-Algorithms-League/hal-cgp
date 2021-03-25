"""
Example for evolutionary regression with multiple genomes
=========================================================

Example demonstrating the use of Cartesian genetic programming with multiple
genomes per individual for a simple regression task with a piecewise
target function.
"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
   Usage:
     example_multi_genome.py [--max-generations=<N>]

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
# We first define a target function.  The function applies different
# transformations to the input depending whether the input is less or greater
# than or equal to zero. We thus need to fit two different functions.


def f_target(x):
    return np.select([x < 0, x >= 0], [-x, x ** 2 + 1.0])


# %%
# Then we define the objective function for the evolution. It uses the
# mean-squared error between the output of the expression represented by a given
# individual and the target function evaluated on a set of random points. We
# here either evaluate the function represented by the first (``f[0]``) or the second
# genome (``f[1]``), depending whether the input is less or greater than zero.


def objective(individual):
    if not individual.fitness_is_None():
        return individual

    n_function_evaluations = 1000

    np.random.seed(1234)

    # Note that f is now a list of functions because individual is an instance
    # of `InvidividualMultiGenome`
    f = individual.to_numpy()
    x = np.random.uniform(-4, 4, (n_function_evaluations, 1))
    y = np.piecewise(x, [x[:, 0] < 0, x[:, 0] >= 0], f)[:, 0]
    loss = np.sum((f_target(x[:, 0]) - y) ** 2)
    individual.fitness = -loss / n_function_evaluations
    return individual


# %%
# Next, we set up the evolutionary search. First, we define the parameters for
# the population, the genomes of individuals, and the evolutionary
# algorithm. Note that we define ``genome_params`` as a list of parameter
# dictionaries which causes the population to create instances of
# ``InvidividualMultiGenome``.

population_params = {"n_parents": 1, "seed": 8188211}

single_genome_params = {
    "n_inputs": 1,
    "n_outputs": 1,
    "n_columns": 12,
    "n_rows": 1,
    "levels_back": 5,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),
}
genome_params = [single_genome_params, single_genome_params]

ea_params = {"n_offsprings": 4, "mutation_rate": 0.03, "n_processes": 1}

evolve_params = {"max_generations": int(args["--max-generations"]), "termination_fitness": 0.0}

# %%
# We create a population that will be evolved
pop = cgp.Population(**population_params, genome_params=genome_params)

# %%
# and an instance of the (mu + lambda) evolutionary algorithm
ea = cgp.ea.MuPlusLambda(**ea_params)

# %%
# We define a callback for recording of fitness over generations
history = {}
history["fitness_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)


# %%
# and finally perform the evolution
cgp.evolve(pop, objective, ea, **evolve_params, print_progress=True, callback=recording_callback)


# %%
# After finishing the evolution, we print the evolved expression and plot the result.
expr = pop.champion.to_sympy()
print(expr)
print(f"--> x<=0: {expr[0]}, \n    x> 0: {expr[1]}")

width = 9.0
fig, axes = plt.subplots(1, 2, figsize=(width, width / scipy.constants.golden))

ax_fitness, ax_function = axes[0], axes[1]
ax_fitness.set_xlabel("Generation")
ax_fitness.set_ylabel("Fitness")

ax_fitness.plot(history["fitness_champion"], label="Champion")

ax_fitness.set_yscale("symlog")
ax_fitness.set_ylim(-1.0e2, 0.1)
ax_fitness.axhline(0.0, color="0.7")

f = pop.champion.to_numpy()
x = np.linspace(-5.0, 5.0, 20)[:, np.newaxis]

y = np.piecewise(x, [x[:, 0] < 0, x[:, 0] >= 0], f)[:, 0]
y_target = f_target(x[:, 0])

ax_function.plot(x, y_target, lw=2, alpha=0.5, label="Target")
ax_function.plot(x, y, "x", label="Champion")
ax_function.legend()
ax_function.set_ylabel(r"$f(x)$")
ax_function.set_xlabel(r"$x$")

fig.savefig("example_multi_genome.pdf", dpi=300)
