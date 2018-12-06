import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import sympy
from sympy.printing.dot import dotprint
import sys
import time
import torch

sys.path.insert(0, '../')
import gp


def objective(individual):

    if individual.fitness is not None:
        return individual

    n_function_evaluations = 1000

    graph = gp.CGPGraph(individual.genome)
    f_graph = graph.compile_torch_class()

    def f_target(x):  # target function
        # return 2.7182 + x[:, 0] - x[:, 1]
        # return (1. + x[:, 0]) * (1. + x[:, 1])
        return 1. / (1. + x[:, 0]) + 1. / (1. + x[:, 1])
        # return 1. / (1. + 1. / x[:, 0]) + 1. / (1. + 1. / x[:, 1])
        # return 1. / (1. + 1. / x[:, 0] ** 2) + 1.
        # return 1. / (1. + 1. / x[:, 0] ** 2) + 1. / (1. + 1. / x[:, 1] ** 2)  # solved, ~50k
        # return 1. / (1. + 1. / x[:, 0] ** 4) + 1. / (1. + 1. / x[:, 1] ** 4)
        # return 1. / (1. + 1. / x[:, 0] ** 2)
        # return x[:, 0] ** 2 + 2 * x[:, 0] * x[:, 1] + x[:, 1] ** 2  # solved, ~3k

    x = torch.Tensor(n_function_evaluations, 2).uniform_(-5, 5)
    x[torch.abs(x) < 1e-5] = 1.  # avoid infinities due to zero division
    y = f_graph(x)

    loss = torch.mean((f_target(x) - y[:, 0]) ** 2)

    individual.fitness = -loss.item()

    return individual


def evo_regression():
    params = {
        'seed': 8188212,
        'n_threads': 1,

        # evo parameters
        'n_parents': 1,
        'n_offspring': 4,
        'max_generations': 1000,
        'min_fitness': 0.,
        'n_breeding': 4,
        'tournament_size': 1,
        'mutation_rate': 0.02,

        # cgp parameters
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 10,
        'n_rows': 5,
        'levels_back': 10,
    }

    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPDiv, gp.CGPConstantFloat])

    # create population object that will be evolved
    pop = gp.CGPPopulation(
        params['n_parents'], params['n_offspring'], params['n_breeding'], params['tournament_size'], params['mutation_rate'],
        params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives, n_threads=params['n_threads'])

    history_fitness, history_average_phenotype_distance, history_average_genotype_distance = pop.evolve(objective, params['max_generations'], params['min_fitness'])

    history_fitness = np.array(history_fitness)

    mean = np.mean(history_fitness, axis=1)
    # std = np.std(history_fitness, axis=1)

    width = 4.
    fig = plt.figure(figsize=(width, width / scipy.constants.golden))
    ax_fitness = fig.add_subplot(311)
    ax_fitness.set_xlabel('Evolution step')
    ax_fitness.set_ylabel('Fitness')

    ax_phenotype_distance = fig.add_subplot(312)
    ax_genotype_distance = fig.add_subplot(313)

    ax_fitness.plot(mean, lw=2, color='k', label='mean')
    # plt.plot(mean + std, color='k', ls='--')
    # plt.plot(mean - std, color='k', ls='--')
    ax_fitness.plot(history_fitness)
    ax_fitness.legend(fontsize=8)

    ax_phenotype_distance.plot(history_average_phenotype_distance)

    ax_genotype_distance.plot(history_average_genotype_distance)

    fig.savefig('example_evo_regression.pdf')

    graph = gp.CGPGraph(pop.champion.genome)
    sympy_expr = graph.compile_sympy_expression()
    print(graph.pretty_print())
    print('evolved expression:', sympy_expr[0])
    print('evolved expression (simplified):', sympy_expr[0].simplify())
    # sympy.plot(sympy_expr[0])

    # export computational graphs; create pdf with
    # $ dot -Tpdf {filename} -O
    with open('example_evo_regression-graph.dot', 'w') as f:
        f.write(dotprint(sympy_expr[0]))
    with open('example_evo_regression-graph_simplified.dot', 'w') as f:
        f.write(dotprint(sympy_expr[0].simplify()))


if __name__ == '__main__':
    evo_regression()
