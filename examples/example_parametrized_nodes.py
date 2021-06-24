"""
Example for evolutionary regression with parametrized nodes
===========================================================

Example demonstrating the use of Cartesian genetic programming for a
regression task that requires fine tuning of constants in parametrized
nodes. This is achieved by introducing a new node, "ParametrizedAdd"
which produces a scaled and shifted version of the sum of its inputs.

"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
  Usage:
    example_parametrized_nodes.py [--max-generations=<N>]

  Options:
    -h --help
    --max-generations=<N>  Maximum number of generations [default: 500]
"""

import functools
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import torch
from docopt import docopt

import cgp

args = docopt(docopt_str)

# %%
# We first define a new node that adds the values of its two inputs then scales and
# finally shifts the result. The scale ("w") and shift factors ("b")
# are parameters that are adapted by local search. We need to define
# the arity of the node, callables for the initial values for the
# parameters and the operation of the node as a string. In this string
# parameters are enclosed in angle brackets, inputs are denoted by "x_i"
# with i representing their corresponding index.


class ParametrizedAdd(cgp.OperatorNode):
    """A node that adds its two inputs.

    The result of addition is scaled by w and shifted by b. Both these
    parameters can be adapted via local search are passed on from
    parents to their offspring.

    """

    _arity = 2
    _initial_values = {"<w>": lambda: 1.0, "<b>": lambda: 0.0}
    _def_output = "<w> * (x_0 + x_1) + <b>"


# %%
# We define a target function which contains numerical constants that
# are not available as constants for the search and need to be found
# by local search on parameterized nodes.


def f_target(x):
    return math.pi * (x[:, 0] + x[:, 1]) + math.e


# %%
# Then we define a differentiable(!) inner objective function for the
# evolution. This function accepts a torch class as a parameter. It
# returns the mean-squared error between the output of the forward
# pass of this class and the target function evaluated on a set of
# random points. This inner objective is then used by actual objective
# function to determine the fitness of the individual.


def inner_objective(f, seed):
    torch.manual_seed(seed)
    batch_size = 500
    x = torch.DoubleTensor(batch_size, 2).uniform_(-5, 5)
    y = f(x)
    return torch.nn.MSELoss()(f_target(x), y[:, 0])


def objective(individual, seed):
    if not individual.fitness_is_None():
        return individual

    f = individual.to_torch()
    loss = inner_objective(f, seed)
    individual.fitness = -loss.item()

    return individual


# %%
# Next, we define the parameters for the population, the genome of
# individuals, the evolutionary algorithm, and the local search. Note
# that we add the custom node defined above as a primitive.

population_params = {"n_parents": 1, "seed": 818821}

genome_params = {
    "n_inputs": 2,
    "n_outputs": 1,
    "n_columns": 5,
    "n_rows": 1,
    "levels_back": None,
    "primitives": (ParametrizedAdd, cgp.Add, cgp.Sub, cgp.Mul),
}

ea_params = {"n_offsprings": 4, "tournament_size": 1, "mutation_rate": 0.04, "n_processes": 2}

evolve_params = {"max_generations": int(args["--max-generations"]), "termination_fitness": 0.0}

local_search_params = {"lr": 1e-3, "gradient_steps": 9}

# %%
# We then create a Population instance and instantiate the local search
# and evolutionary algorithm.

pop = cgp.Population(**population_params, genome_params=genome_params)

local_search = functools.partial(
    cgp.local_search.gradient_based,
    objective=functools.partial(inner_objective, seed=population_params["seed"]),
    **local_search_params,
)

ea = cgp.ea.MuPlusLambda(**ea_params, local_search=local_search)


# %%
# We define a recording callback closure for bookkeeping of the progression of the evolution.

history = {}
history["fitness_champion"] = []
history["expr_champion"] = []


def recording_callback(pop):
    history["fitness_champion"].append(pop.champion.fitness)
    history["expr_champion"].append(pop.champion.to_sympy())


# %%
# We fix the seed for the objective function to make sure results are
# comparable across individuals and, finally, we call the `evolve`
# method to perform the evolutionary search.

obj = functools.partial(objective, seed=population_params["seed"])

cgp.evolve(obj, pop, ea, **evolve_params, print_progress=True, callback=recording_callback)

# %%
# After the evolutionary search has ended, we print the expression
# with the highest fitness and plot the progression of the search.

print(f"Final expression {pop.champion.to_sympy()} with fitness {pop.champion.fitness}")

print("Best performing expression per generation (for fitness increase > 0.5):")
old_fitness = -np.inf
for i, (fitness, expr) in enumerate(zip(history["fitness_champion"], history["expr_champion"])):
    delta_fitness = fitness - old_fitness
    if delta_fitness > 0.5:
        print(f"{i:3d}: {fitness}, {expr}")
        old_fitness = fitness
print(f"{i:3d}: {fitness}, {expr}")

width = 9.0

fig = plt.figure(figsize=(width, width / scipy.constants.golden))

ax_fitness = fig.add_subplot(111)
ax_fitness.set_xlabel("Generation")
ax_fitness.set_ylabel("Fitness")
ax_fitness.set_yscale("symlog")

ax_fitness.axhline(0.0, color="k")
ax_fitness.plot(history["fitness_champion"], lw=2)

plt.savefig("example_parametrized_nodes.pdf", dpi=300)
