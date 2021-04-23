"""
Example for evolutionary regression on a piecewise target function
==================================================================

Example demonstrating the use of Cartesian genetic programming for
regression on a piecewise target function by using the conditional (if/else) operator.
"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
   Usage:
     example_piecewise_target_function.py [--max-generations=<N>]

   Options:
     -h --help
     --max-generations=<N>  Maximum number of generations [default: 2000]
"""

import functools

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

import cgp

args = docopt(docopt_str)


# %%
# We define a piecewise target function.  The function applies different
# transformations to the input depending whether the input is less or greater
# than zero. Thus to achieve high fitness,
# an individual must make use of the if/else operator.
def f_target(x):
    return np.select([x >= 0, x < 0], [x ** 2 + 1.0, -x])


# %%
# Then we define the objective function for the evolution. It uses the
# mean-squared error between the output of the expression represented by a given
# individual and the target function evaluated on a set of pseudo-random points.
def objective(individual, rng):
    """Objective function of the regression task.

    Parameters
    ----------
    individual : Individual
        Individual of the Cartesian Genetic Programming Framework.
    rng: numpy.random.RandomState

    Returns
    -------
    Individual
        Modified individual with updated fitness value.
    """
    if not individual.fitness_is_None():
        return individual

    n_function_evaluations = 1000

    f = individual.to_numpy()
    x = rng.uniform(-5, 5, size=n_function_evaluations)
    y = f(x)

    loss = np.mean((f_target(x) - y) ** 2)
    individual.fitness = -loss

    return individual


# %%
# Next, we set up the evolutionary search. First, we define the parameters for
# the population, the genomes of individuals, and the evolutionary
# algorithm.
population_params = {
    "n_parents": 1,
    "seed": 8188211,
}

genome_params = {
    "n_inputs": 1,
    "n_outputs": 1,
    "n_columns": 20,
    "n_rows": 1,
    "levels_back": None,
    "primitives": (cgp.IfElse, cgp.Mul, cgp.Add, cgp.Sub, cgp.ConstantFloat,),
}

ea_params = {"n_offsprings": 4, "mutation_rate": 0.03, "n_processes": 2}

evolve_params = {"max_generations": int(args["--max-generations"]), "termination_fitness": 0.0}

# create population that will be evolved
pop = cgp.Population(**population_params, genome_params=genome_params)

# create instance of evolutionary algorithm
ea = cgp.ea.MuPlusLambda(**ea_params)

# define callback for recording of fitness over generations
history = {}
history["fitness_champion"] = []
history["expr_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)
    try:
        sympy_expression = pop.champion.to_sympy()
    except TypeError:
        sympy_expression = pop.champion.to_sympy(simplify=False)

    history["expr_champion"].append(sympy_expression)


# the objective passed to evolve should only accept one argument,
# the individual
rng = np.random.RandomState(seed=population_params["seed"])
obj = functools.partial(objective, rng=rng)

# Perform the evolution
cgp.evolve(pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback)

# %%
# After the evolutionary search has ended, we print the expression
# with the highest fitness and plot the search progression and target and evolved functions.
print(f"Final expression {pop.champion.to_sympy()} with fitness {pop.champion.fitness}")

fig = plt.figure(1)
plt.plot(history["fitness_champion"])
plt.ylim(1.1 * min(history["fitness_champion"]), 5)
plt.xlabel("Generation")
plt.ylabel("Loss (Fitness)")
plt.legend(["Champion loss per generation"])
plt.title({pop.champion.to_sympy()})
fig.savefig("example_piecewise_fitness_history.pdf")

x = np.arange(-5, 5, 0.01)
champion_numpy = pop.champion.to_numpy()

fig = plt.figure(2)
plt.subplot(121)
plt.plot(x, f_target(x), "b")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Target function")
plt.legend(["target"])
plt.subplot(122)
plt.plot(x, champion_numpy(x), "r")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Evolved function")
plt.legend(["champion"])
fig.savefig("example_piecewise_target_function.pdf", dpi=300)
