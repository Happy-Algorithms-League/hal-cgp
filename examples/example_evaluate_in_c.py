"""
Example for evolutionary regression, with evaluation in cpp
===========================================
"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
   Usage:
     example_evaluate_in_c.py 

   Options:
     -h --help
"""

import ctypes, ctypes.util
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import subprocess
from docopt import docopt

import cgp

args = docopt(docopt_str)

# %%
# Then we define the objective function for the evolution. It uses
# the mean-squared error between the output of the expression
# represented by a given individual and the target function evaluated
# on a set of random points.


def objective(individual):

    if not individual.fitness_is_None():
        return individual
    #individual = set_initial_dna(individual)  # todo remove (debugging stuff)

    graph = cgp.CartesianGraph(individual.genome)
    function_name = 'rule'
    filename = 'individual'
    path = 'c_code'

    # todo: combine filename with individual id? f'individual_{individual.idx}'? - Issue in the main.c import
    graph.to_c(function_name=function_name, filename=filename, path=path)  #

    def compile_c_code(filename, path):
        subprocess.run(["gcc", "-c", "-fPIC", f"{path}/{filename}.c", "-o", f"{path}/{filename}.o", ])  # todo: catch errors
        subprocess.run(["gcc", "-c", "-fPIC", f"{path}/main.c", "-o", f"{path}/main.o", ])
        # subprocess.run(["gcc", f"{path}/main.o", f"{path}/{filename}.o", "-shared", "-o", f"{path}/{filename}.so"])
        subprocess.run(["gcc", f"{path}/main.o", f"{path}/{filename}.o", "-o", f"{path}/{filename}"])


    # compile_c_code()
    compile_c_code(filename, path)

    #libname = pathlib.Path().absolute() / f"{path}/{filename}.so"
    #c_lib = ctypes.CDLL(libname)
    #c_lib.l2_norm_rule_target.restype = ctypes.c_double  # set output type to double

    # run simulation
    #individual.fitness = -1.0 * c_lib.l2_norm_rule_target()
    individual.fitness = -1.0 * float(subprocess.check_output(pathlib.Path().absolute() / f"{path}/{filename}"))

    return individual


set_solution_initially = False

genome_params = {
    "n_inputs": 2,
    "primitives": (cgp.Add, cgp.Mul, cgp.ConstantFloat)
}

seed = 123456789


# target = x_0 * x_1 + 1.0;
def set_initial_dna(ind):
    genome = cgp.Genome(**genome_params)
    genome.randomize(rng=np.random.RandomState(seed=1234))

    #dna_prior = [1, 0, 1, 2, 0, 0, 0, 2, 3]  # Mul as 2nd operator (1): x_0*x1; 2 as const
    dna_prior = [2,0,0, 2,0,0, 0,2,3]
    genome.set_expression_for_output(dna_insert=dna_prior)
    ind = cgp.IndividualSingleGenome(genome)
    print(ind.to_sympy())
    return cgp.IndividualSingleGenome(genome)


if set_solution_initially:
    pop = cgp.Population(genome_params=genome_params, individual_init=set_initial_dna, seed=seed)
else:
    pop = cgp.Population(genome_params=genome_params, seed=seed)


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
    objective=objective,  pop=pop, termination_fitness=0.0, max_generations=1000,
    print_progress=True, callback=recording_callback
)

print(pop.champion.to_sympy())


# %%
# After finishing the evolution, we plot the result and log the final
# evolved expression.


def plot_champion_and_target(f_champion, f_target):
    width = 9.0
    fig, axes = plt.subplots(1, 2, figsize=(width, width / 1.62))

    ax_fitness, ax_function = axes[0], axes[1]
    ax_fitness.set_xlabel("Generation")
    ax_fitness.set_ylabel("Fitness")

    ax_fitness.plot(history["fitness_champion"], label="Champion")

    ax_fitness.set_yscale("symlog")
    ax_fitness.set_ylim(-1.0e2, 0.1)
    ax_fitness.axhline(0.0, color="0.7")

    x = np.linspace(-5.0, 5.0, 20)
    y = [f_champion(x_i) for x_i in x]
    y_target = [f_target(x_i) for x_i in x]

    ax_function.plot(x, y_target, lw=2, alpha=0.5, label="Target")
    ax_function.plot(x, y, "x", label="Champion")
    ax_function.legend()
    ax_function.set_ylabel(r"$f(x)$")
    ax_function.set_xlabel(r"$x$")

    fig.savefig("example_evaluate_in_cpp.pdf", dpi=300)


# plot_champion_and_target(f_champion=pop.champion.to_func, f_target=f_target)
