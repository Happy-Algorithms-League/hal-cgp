"""
Evolving boolean expressions
============================

Example demonstrating the use of Cartesian genetic programming for
generating boolean expressions from a truth table.
"""

# The docopt str is added explicitly to ensure compatibility with
# sphinx-gallery.
docopt_str = """
   Usage:
     example_parity.py [--max-generations=<N>]

   Options:
     -h --help
     --max-generations=<N>  Maximum number of generations [default: 300]
"""

from docopt import docopt

import cgp

args = docopt(docopt_str)

# %%
# We first define a truth table (here 3bit parity generator).

truth_table = {
    (0, 0, 0): 0,
    (0, 0, 1): 1,
    (0, 1, 0): 1,
    (0, 1, 1): 0,
    (1, 0, 0): 1,
    (1, 0, 1): 0,
    (1, 1, 0): 0,
    (1, 1, 1): 1,
}


# %%
# Then we define the objective function for the evolution. It check whether the
# output of our expression matches the expected value for all input
# combinations.


def objective(individual):

    if not individual.fitness_is_None():
        return individual

    f = individual.to_func()
    fitness = 0
    for message, parity_bit in truth_table.items():
        y = f(*message)
        fitness += float(y == bool(parity_bit))

    individual.fitness = fitness

    return individual


class AND2(cgp.OperatorNode):
    """A node that ands its two inputs."""

    _arity = 2
    _def_output = "bool(x_0) and bool(x_1)"
    _def_numpy_output = "np.logical_and(x_0.astype(bool), x_1.astype(bool))"
    _def_sympy_output = "x_0 & x_1"
    _def_torch_output = "torch.logical_and(x_0.bool(), x_1.bool())"


class OR2(cgp.OperatorNode):
    """A node that ors its two inputs."""

    _arity = 2
    _def_output = "bool(x_0) or bool(x_1)"
    _def_numpy_output = "np.logical_or(x_0.astype(bool), x_1.astype(bool))"
    _def_sympy_output = "x_0 | x_1"
    _def_torch_output = "torch.logical_or(x_0.bool(), x_1.bool())"


class NOT(cgp.OperatorNode):
    """A node that nots its input."""

    _arity = 1
    _def_output = "not bool(x_0)"
    _def_numpy_output = "np.logical_not(x_0.astype(bool))"
    _def_sympy_output = "Not(x_0)"
    _def_torch_output = "torch.logical_not(x_0.bool())"


class XOR2(cgp.OperatorNode):
    """A node that xors its two inputs."""

    _arity = 2
    _def_output = "bool(x_0) != bool(x_1)"
    _def_numpy_output = "np.logical_xor(x_0.astype(bool), x_1.astype(bool))"
    _def_sympy_output = "Xor(x_0, x_1)"
    _def_torch_output = "torch.logical_xor(x_0.bool(), x_1.bool())"


genome_params = {
    "n_inputs": 3,
    "primitives": (AND2, OR2, NOT, XOR2),
}

# create population that will be evolved
pop = cgp.Population(genome_params=genome_params)


# %%
# Next, we perform the evolution mostly relying on the libraries default
# hyperparameters.
pop = cgp.evolve(
    objective,
    pop,
    termination_fitness=8.0,  # eight rows in truth table, so max fitness is 8
    print_progress=True,
)


# %%
# After finishing the evolution, we log the final evolved expression.
f = pop.champion.to_sympy(simplify=True)
print(f"Final expression: {f}")
