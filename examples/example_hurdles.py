"""
Minimal example for evolutionary regression using hurdles
=========================================================

Example demonstrating the use of Cartesian genetic programming for a
simple regression task where we use hurdles to implement early
stopping for low-performing invididuals.

Hurdles are implemented by introducing multiple objectives, here two,
which are sequentially evaluated. Only those individuals with fitness
in the upper 50th percentile on the first objective are evaluated on
the second objective.

References:

- Real, E., Liang, C., So, D., & Le, Q. (2020, November). AutoML-zero:
  evolving machine learning algorithms from scratch. In International
  Conference on Machine Learning (pp. 8007-8019). PMLR.

"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
   Usage:
     example_minimal.py [--max-generations=<N>]

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
# Then we define two objective functions for the evolution. They use
# the mean-squared error between the output of the expression
# represented by a given individual and the target function evaluated
# on a set of random points. The first objective uses only few samples
# (100) to get a fast estimate how well an individual performs. The
# second objective uses may samples (99900) to determine the fitness
# precisely.


def objective_one(individual):

    if not individual.fitness_is_None():
        return individual

    n_function_evaluations = 100000

    np.random.seed(1234)
    values = np.random.uniform(-4, 4, n_function_evaluations)

    f = individual.to_func()
    loss = 0
    for x in values[:100]:
        # the callable returned from `to_func` accepts and returns
        # lists; accordingly we need to pack the argument and unpack
        # the return value
        y = f(x)
        loss += (f_target(x) - y) ** 2

    individual.fitness = -loss / n_function_evaluations

    return individual


def objective_two(individual):

    if not individual.fitness_is_None():
        return individual

    n_function_evaluations = 100000

    np.random.seed(1234)
    values = np.random.uniform(-4, 4, n_function_evaluations)

    f = individual.to_func()
    loss = 0
    for x in values[100:]:
        # the callable returned from `to_func` accepts and returns
        # lists; accordingly we need to pack the argument and unpack
        # the return value
        y = f(x)
        loss += (f_target(x) - y) ** 2

    individual.fitness = -loss / n_function_evaluations

    return individual


# %%
# Next, we set up the evolutionary search. We first define the
# parameters for the population, the genome of individuals, and the
# evolutionary algorithm.


population_params = {"n_parents": 1, "seed": 8188211}

genome_params = {
    "n_inputs": 1,
    "n_outputs": 1,
    "n_columns": 12,
    "n_rows": 1,
    "levels_back": 5,
    "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),
}

# %%
# We define the upper percentile of individuals which are evaluated on
# the (n+1)th objective by a list of numbers between 0 and 1.
ea_params = {
    "n_offsprings": 4,
    "mutation_rate": 0.03,
    "n_processes": 1,
    "hurdle_percentile": [0.5, 0.0],
}

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
cgp.evolve(
    [objective_one, objective_two],
    pop,
    ea,
    **evolve_params,
    print_progress=True,
    callback=recording_callback
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

fig.savefig("example_minimal.pdf", dpi=300)
