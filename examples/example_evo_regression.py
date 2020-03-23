import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.constants
import functools
import sys
import torch

from sympy.printing.dot import dotprint

sys.path.insert(0, "../")
import gp

"""
Example script showing how to use Cartesian Genetic Programming for
a simple regression task.
"""


def f_target_easy(x):  # target function
    return x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1] + x[:, 1] ** 2


def f_target_hard(x):  # target function
    return 1.0 / (1.0 + 1.0 / x[:, 0] ** 2) + 1.0 / (1.0 + 1.0 / x[:, 1] ** 2)


def objective(individual, target_function):
    """Objective function of the regression task.

    Parameters
    ----------
    individual : gp.CGPIndividual
        Individual of the Cartesian Genetic Programming Framework.
    target_function : Callable
        Target function.

    Returns
    -------
    gp.CGPIndividual
        Modified individual with updated fitness value.
    """
    if individual.fitness is not None:
        return individual

    n_function_evaluations = 1000

    graph = gp.CGPGraph(individual.genome)
    f_graph = graph.to_torch()
    x = torch.Tensor(n_function_evaluations, 2).uniform_(-5, 5)
    y = f_graph(x)
    loss = torch.mean((target_function(x) - y[:, 0]) ** 2)
    individual.fitness = -loss.item()

    return individual


def evolution(f_target):
    """Execute CGP on a regression task for a given target function.

    Parameters
    ----------
    f_target : Callable
        Target function

    Returns
    -------
    dict
        Dictionary containing the history of the evolution
    """
    params = {
        "seed": 8188212,
        "n_threads": 1,
        "max_generations": 1000,
        "min_fitness": 0.0,
        "population_params": {"n_parents": 10, "mutation_rate": 0.5},
        "ea_params": {"n_offsprings": 10, "n_breeding": 10, "tournament_size": 1},
        "genome_params": {
            "n_inputs": 2,
            "n_outputs": 1,
            "n_columns": 10,
            "n_rows": 5,
            "levels_back": 2,
            "primitives": [gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPDiv, gp.CGPConstantFloat],
        },
    }

    np.random.seed(params["seed"])
    torch.manual_seed(params["seed"])

    # create population object that will be evolved
    pop = gp.CGPPopulation(
        **params["population_params"], seed=params["seed"], genome_params=params["genome_params"]
    )

    # create instance of evolutionary algorithm
    ea = gp.ea.MuPlusLambda(**params["ea_params"])

    def record_history(pop, history):
        keys = ["champion", "fitness_parents"]
        for key in keys:
            if key not in history:
                history[key] = []
        history["champion"].append(pop.champion)
        history["fitness_parents"].append(pop.fitness_parents())

    obj = functools.partial(objective, target_function=f_target)
    # Perform evolution
    history = gp.evolve(
        pop,
        obj,
        ea,
        params["max_generations"],
        params["min_fitness"],
        record_history=record_history,
        print_progress=True,
    )
    return history, pop


if __name__ == "__main__":
    width = 9.0
    fig, axes = plt.subplots(2, 2, figsize=(width, width / scipy.constants.golden))

    for i, (label, target_function) in enumerate(
        zip(["easy", "hard"], [f_target_easy, f_target_hard])
    ):
        history, pop = evolution(target_function)

        ax_fitness, ax_function = axes[i]
        ax_fitness.set_xlabel("Generation")
        ax_fitness.set_ylabel("Fitness")

        history_fitness = np.array(history["fitness_parents"])
        ax_fitness.plot(np.max(history_fitness, axis=1), label="Champion")
        ax_fitness.plot(np.mean(history_fitness, axis=1), label="Population mean")

        ax_fitness.set_yscale("symlog")
        ax_fitness.set_ylim(-1.0e4, 0.0)
        ax_fitness.legend()

        # Evaluate final champion
        graph = gp.CGPGraph(pop.champion.genome)
        sympy_expr = graph.to_sympy(simplify=False)
        print(graph.pretty_print())
        print("evolved expression:", sympy_expr[0])
        print("evolved expression simplified:", sympy_expr[0].simplify())

        x0_range = np.arange(-5.0, 5.0, 0.1)
        x1_range = [-2.0, 2.0]

        y = [[sympy_expr[0].subs({"x_0": x0, "x_1": x1}) for x0 in x0_range] for x1 in x1_range]
        y_target = [
            target_function(np.stack((x0_range, np.ones_like(x0_range) * x1)).T) for x1 in x1_range
        ]

        ax_function.plot(x0_range, y_target[0], lw=2, alpha=0.5, label="Target")
        ax_function.plot(x0_range, y[0], "x", label="Champion")
        ax_function.legend()
        ax_function.set_ylabel(r"$f(x)$")
        ax_function.set_xlabel(r"$x$")

        # export computational graphs
        # full graph
        fn = f"example_evo_regression-graph_{label}"
        with open(f"{fn}.dot", "w") as f:
            f.write(dotprint(sympy_expr[0]))
        os.system(f"dot -Tps {fn}.dot -o {fn}.pdf")
        # simplified graph
        fn = f"example_evo_regression-graph_{label}_simplified"
        with open(f"{fn}.dot", "w") as f:
            f.write(dotprint(sympy_expr[0].simplify()))
        os.system(f"dot -Tps {fn}.dot -o {fn}.pdf")

        with open(".dot", "w") as f:
            f.write(dotprint(sympy_expr[0].simplify()))
        os.system(f"dot -Tps {fn}.dot -o {fn}.pdf")

    # plt.tight_layout()
    fig.savefig("example_evo_regression.pdf")
