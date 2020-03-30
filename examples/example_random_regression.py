import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.constants

import gp


def objective(individual):
    if individual.fitness is not None:
        return individual

    n_function_evaluations = 1000

    f_graph = individual.to_torch()
    x = torch.Tensor(n_function_evaluations, 2).uniform_(-5, 5)
    y = f_graph(x)
    loss = torch.mean((f_target(x) - y[:, 0]) ** 2)
    individual.fitness = -loss.item()

    return individual


def evolution():

    params = {
        "population_params": {"n_parents": 10, "mutation_rate": 0.5, "seed": 8188211},
        "genome_params": {
            "n_inputs": 1,
            "n_outputs": 1,
            "n_columns": 10,
            "n_rows": 2,
            "levels_back": 2,
            "primitives": [gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPDiv, gp.CGPParameter],
        },
        "ea_params": {
            "n_offsprings": 10,
            "n_breeding": 10,
            "tournament_size": 1,
            "n_processes": 1,
        },
        "local_search_params": {"lr": 2e-3, "batch_size": 100},
        "evolve_params": {"max_generations": 500, "min_fitness": -1e-12},
    }

    np.random.seed(params["population_params"]["seed"])
    torch.manual_seed(params["population_params"]["seed"])

    pop = gp.CGPPopulation(**params["population_params"], genome_params=params["genome_params"])

    ea = gp.ea.MuPlusLambda(**params["ea_params"])

    def local_search(pop):
        for ind in pop:
            f = ind.to_torch()
            if len(list(f.parameters())) > 0:
                optimizer = torch.optim.SGD(f.parameters(), lr=params["local_search_params"]["lr"])
                criterion = torch.nn.MSELoss()

                for i in range(10):
                    x = torch.Tensor(params["local_search_params"]["batch_size"], 1).normal_()
                    y = f(x)
                    y_target = f_target(x)

                    loss = criterion(y[:, 0], y_target)
                    f.zero_grad()
                    loss.backward()
                    optimizer.step()

                ind.update_parameters_from_torch_class(f)

    ea.local_search = local_search

    history = {}
    history["champion"] = []
    history["fitness_parents"] = []

    def recording_callback(pop):
        history["champion"].append(pop.champion)
        history["fitness_parents"].append(pop.fitness_parents())

    gp.evolve(
        pop,
        objective,
        ea,
        **params["evolve_params"],
        print_progress=True,
        callback=recording_callback,
    )

    return history, pop.champion


def f_target(x):  # target function
    return x[:, 0] ** 2 + 1.0 / np.pi


if __name__ == "__main__":

    width = 9.0

    fig = plt.figure(figsize=(width, width / scipy.constants.golden))

    ax_fitness = fig.add_subplot(121)
    ax_fitness.set_xlabel("Generation")
    ax_fitness.set_ylabel("Fitness")

    ax_function = fig.add_subplot(122)
    ax_function.set_ylabel(r"$f(x)$")
    ax_function.set_xlabel(r"$x$")

    history, champion = evolution()

    history_fitness = np.array(history["fitness_parents"])
    ax_fitness.plot(np.max(history_fitness, axis=1), label="Champion")
    ax_fitness.plot(np.mean(history_fitness, axis=1), label="Population mean")

    x = np.linspace(-5.0, 5, 100).reshape(-1, 1)
    print(champion.to_sympy(simplify=False))
    f = champion.to_func()
    y = [f(xi) for xi in x]
    ax_function.plot(x, f_target(x), lw=2, label="Target")
    ax_function.plot(x, y, lw=1, label="Target", marker="x")

    plt.savefig("example_random_regression.pdf", dpi=300)
