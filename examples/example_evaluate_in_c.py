"""
Example for evolutionary regression, with evaluation in C
=========================================================
"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
   Usage:
     example_evaluate_in_c.py

   Options:
     -h --help
"""
import functools
import pathlib
import subprocess
from docopt import docopt

import cgp

args = docopt(docopt_str)

# %%
# We first define a helper function for compiling the C code. It creates
# object files from the file and main script and creates an executable


def compile_c_code(path):

    # assert all necessary files exist
    path_file_c = pathlib.Path(f"{path}/individual.c")
    path_file_h = pathlib.Path(f"{path}/individual.h")
    path_script_o = pathlib.Path(f"{path}/main.o")
    assert path_file_c.is_file() & path_file_h.is_file() & path_script_o.is_file()

    # compile file with rule
    subprocess.check_call(
        ["gcc", "-c", "-fPIC", f"{path}/individual.c", "-o", f"{path}/individual.o"]
    )

    # create executable
    subprocess.check_call(
        ["gcc", f"{path}/main.o", f"{path}/individual.o", "-o", f"{path}/individual"]
    )


# %%
# We define the objective function for the evolution. It creates a
# C module and header from the computational graph. File with rule
# and script for evaluation are compiled using the above helper function.
# It assigns fitness to the negative float of the print of the script execution.


def objective(individual, path):

    if not individual.fitness_is_None():
        return individual

    graph = cgp.CartesianGraph(individual.genome)

    graph.to_c(path=path)

    compile_c_code(path=path)

    result = subprocess.check_output(pathlib.Path().absolute() / f"{path}/individual")
    assert result

    individual.fitness = -1.0 * float(result)

    return individual


# %%
# Next, we set up the evolutionary search. We first define the parameters of the
# genome. We then create a population of individuals with matching genome parameters.


genome_params = {"n_inputs": 2, "primitives": (cgp.Add, cgp.Mul, cgp.ConstantFloat)}

pop = cgp.Population(genome_params=genome_params)

# compile C script
path = "c_code"
assert pathlib.Path(f"{path}/main.c")
subprocess.check_call(["gcc", "-c", "-fPIC", f"{path}/main.c", "-o", f"{path}/main.o"])

# the objective passed to evolve should only accept one argument,
# the individual
obj = functools.partial(objective, path=path)

# %%
# and finally perform the evolution relying on the libraries default
# hyperparameters except that we terminate the evolution as soon as one
# individual has reached fitness zero.
pop = cgp.evolve(objective=obj, pop=pop, termination_fitness=0.0, print_progress=True)

# %%
# After finishing the evolution, we print the final evolved expression and assert it is the target expression.
print(pop.champion.to_sympy())
assert str(pop.champion.to_sympy()) == "x_0*x_1 + 1.0"
