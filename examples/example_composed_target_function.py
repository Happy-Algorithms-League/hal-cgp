"""
Example for evolutionary regression on a composed target function
==================================================================

Example demonstrating the use of Cartesian genetic programming for
regression on a target function composed of two different sub-functions.
Demonstrates the use of comparison operator genes (>, <) and if/else branching genes.
"""

import functools
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants
import warnings
import sympy

import cgp


# Define new operator genes
class ConstantFloatZero(cgp.ConstantFloat):
    # Should be useful for comparison operator (given the target function structure)
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 0


class ComparisonOperator(cgp.OperatorNode):
    """A node that compares its inputs and outputs 1.0 if the first is bigger and - 1.0 if the second is bigger """

    _arity = 2
    _def_output = "1.0 * ((x_0 - x_1)/max(abs(x_0 - x_1), 0.000001))"
    _def_torch_output = "1.0 * torch.sign(x_0 - x_1)"
    _def_numpy_output = "1.0 * np.sign(x_0 - x_1)"


class IfElseOperator(cgp.OperatorNode):
    """A node that checks its input0, and outputs input1 if true and input2 otherwise """

    _arity = 3

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        if inputs[0]:
            self._output = inputs[1]
        else:
            self._output = inputs[2]

    _def_output = "x_1 if x_0 >= 0 else x_2"
    _def_numpy_output = "np.piecewise(x_0, [x_0>= 0, x_0<0], [x_1 [x_0>= 0] , x_2[x_0<0]])"
    _def_sympy_output = "Piecewise((x_1, x_0 >= 0), (x_2, x < 0), (g, True))"
    _def_torch_output = "torch.where(x_0 >= 0, x_1, x_2)"


def f_target(x):
    c = 3
    y = np.zeros_like(x)
    y[x < 0] = -x[x <= 0]  # + c
    y[x > 0] = x[x > 0] ** 2
    return y


def f_target_simple(x):
    y = np.zeros_like(x)
    y[x <= 0] = x[x <= 0] ** 2
    y[x > 0] = x[x > 0]
    return y


def objective(individual, target_function, seed):
    """Objective function of the regression task.

    Parameters
    ----------
    individual : Individual
        Individual of the Cartesian Genetic Programming Framework.
    target_function : Callable
        Target function.

    Returns
    -------
    Individual
        Modified individual with updated fitness value.
    """
    if individual.fitness is not None:
        return individual

    n_function_evaluations = 1000

    np.random.seed(seed)

    f_graph = individual.to_func()
    y = np.empty(n_function_evaluations)
    x = np.random.uniform(-5, 5, size=(n_function_evaluations, 1))
    for i, x_i in enumerate(x):
        with warnings.catch_warnings():  # ignore warnings due to zero division
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in double_scalars"
            )
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in double_scalars"
            )

            y[i] = f_graph([x_i, 0])[0]

    loss = np.mean((target_function(x) - y) ** 2)
    individual.fitness = -loss

    return individual


def evolution(f_target):
    """Execute CGP on a regression task for a target function composed of two sub-functions.

    Parameters
    ----------
    f_target : Callable
        Target function

    Returns
    -------
    dict
        Dictionary containing the history of the evolution
        with parents fitness, dna of fittest individual in every generation
    Individual
        Individual with the highest fitness in the last generation
    """
    population_params = {"n_parents": 1, "mutation_rate": 0.005, "seed": 8188211}

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 1000,
        "n_rows": 1,
        "levels_back": None,
        "primitives": (
            #cgp.Add, -> Add not required for target function
            #cgp.Sub,  # needed (0-x) if x<0 (not needed if using f_target_simple)
            cgp.Mul, # needed for x² if x>0
            #cgp.Div,-> Div not required for target function
            #ConstantFloatZero,  # needed for comparison operator
            ComparisonOperator,
            IfElseOperator,
        ),
    }

    ea_params = {"n_offsprings": 4, "tournament_size": 2, "n_processes": 2}

    evolve_params = {"max_generations": 100, "min_fitness": 0.0}

    # create population that will be evolved
    pop = cgp.Population(**population_params, genome_params=genome_params)

    # create instance of evolutionary algorithm
    ea = cgp.ea.MuPlusLambda(**ea_params)

    # define callback for recording of fitness over generations
    history = {}
    history["fitness_parents"] = []
    history["champion_dna"] = []

    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())
        history["champion_dna"].append(pop.champion.genome.dna)

    # the objective passed to evolve should only accept one argument,
    # the individual
    obj = functools.partial(objective, target_function=f_target, seed=population_params["seed"])

    # Perform the evolution
    cgp.evolve(pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback)
    return history, pop.champion


if __name__ == "__main__":
    history, champion = evolution(f_target_simple)
    champion_sympy_expression = champion.to_sympy()
    a = 1