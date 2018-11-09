import numpy as np
import matplotlib.pyplot as plt
import sys
import torch

sys.path.insert(0, '../')
import gp


def evo_regression():
    params = {
        'seed': 81882,

        # evo parameters
        'n_individuals': 5,
        'generations': 1000,
        'n_breeding': 5,
        'tournament_size': 6,
        'n_mutations': 3,

        # cgp parameters
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 3,
        'n_rows': 3,
        'levels_back': 2,
    }

    np.random.seed(params['seed'])

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPConstantFloat])

    def objective(genome):

        n_function_evaluations = 100

        graph = gp.CGPGraph(genome)
        f_graph = graph.compile_torch_class()

        def f_target(x):  # target function
            # return 2.7182 + x[0] - x[1]
            return 1. + x[0] - x[1]

        loss = 0.
        for j in range(n_function_evaluations):
            x = torch.Tensor(2).normal_()
            y = f_graph(x)

            loss += (f_target(x) - y[0]) ** 2

        return -1. / n_function_evaluations * loss

    # create population object that will be evolved
    pop = gp.CGPPopulation(
        params['n_individuals'], params['n_breeding'], params['tournament_size'], params['n_mutations'],
        params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)

    # generate initial parent population of size N
    pop.generate_random_parent_population()

    # generate initial offspring population of size N
    pop.generate_random_offspring_population()

    # perform evolution
    history_fitness = []
    for i in range(params['generations']):
        # combine parent and offspring populations and compute fitness for
        # all objectives for all individuals
        pop.compute_fitness(objective)

        # sort population according to fitness & crowding distance
        pop.sort()

        # fill new parent population according to sorting
        pop.create_new_parent_population()

        # generate new offspring population from parent population
        pop.create_new_offspring_population()

        # perform local search to tune values of constants
        # TODO pop.local_search(objective)

        history_fitness.append(pop.fitness)

    plt.plot(np.mean(history_fitness, axis=1))
    plt.show()


if __name__ == '__main__':
    evo_regression()
