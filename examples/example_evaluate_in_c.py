"""
Example for evolutionary regression, with evaluation in c
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

import pathlib
import subprocess
from docopt import docopt

import cgp

args = docopt(docopt_str)

# %%
# We first define a helper function for compiling the c code. It creates
# object files from the file and main script and creates an executable


def compile_c_code(filename, scriptname, path):

    # assert all necessary files exist
    path_file_c = pathlib.Path(f"{path}/{filename}.c")
    path_file_h = pathlib.Path(f"{path}/{filename}.h")
    path_script_c = pathlib.Path(f"{path}/{scriptname}.c")
    path_script_h = pathlib.Path(f"{path}/{scriptname}.h")
    assert (
        path_file_c.is_file()
        & path_file_h.is_file()
        & path_script_c.is_file()
        & path_script_h.is_file()
    )

    # compile file with rule
    subprocess.run(["gcc", "-c", "-fPIC", f"{path}/{filename}.c", "-o", f"{path}/{filename}.o"])
    # compile script
    subprocess.run(
        ["gcc", "-c", "-fPIC", f"{path}/{scriptname}.c", "-o", f"{path}/{scriptname}.o"]
    )
    # create executable
    subprocess.run(
        ["gcc", f"{path}/{scriptname}.o", f"{path}/{filename}.o", "-o", f"{path}/{filename}"]
    )


# %%
# We define the objective function for the evolution. It creates a
# c module and header from the computational graph. File with rule
# and script for evaluation are compiled using the above helper function.
# It assigns fitness to the negative float of the print of the script execution.


def objective(individual):

    if not individual.fitness_is_None():
        return individual

    graph = cgp.CartesianGraph(individual.genome)
    function_name = "rule"
    filename = "individual"
    scriptname = "main"
    path = "c_code"

    graph.to_c(function_name=function_name, filename=filename, path=path)

    # compile_c_code()
    compile_c_code(filename=filename, scriptname=scriptname, path=path)

    # assert that the executable returns something
    assert subprocess.check_output(pathlib.Path().absolute() / f"{path}/{filename}")
    # run simulation and assign fitness
    individual.fitness = -1.0 * float(
        subprocess.check_output(pathlib.Path().absolute() / f"{path}/{filename}")
    )

    return individual


# %%
# Next, we set up the evolutionary search. We first define the parameters of the
# genome. We then create a population of individuals with matching genome parameters.


genome_params = {"n_inputs": 2, "primitives": (cgp.Add, cgp.Mul, cgp.ConstantFloat)}

pop = cgp.Population(genome_params=genome_params)


# %%
# and finally perform the evolution relying on the libraries default
# hyperparameters except that we terminate the evolution as soon as one
# individual has reached fitness zero.

pop = cgp.evolve(objective=objective, pop=pop, termination_fitness=0.0, print_progress=True)

# %%
# After finishing the evolution, we print the final evolved expression.
print(pop.champion.to_sympy())
