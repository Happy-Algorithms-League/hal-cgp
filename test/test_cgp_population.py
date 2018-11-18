import numpy as np
import sys
import torch

sys.path.insert(0, '../')
import gp

SEED = np.random.randint(2 ** 31)


def test_cgp_population():

    params = {
        # evo parameters
        'n_parents': 5,
        'n_offsprings': 5,
        'generations': 500,
        'n_breeding': 5,
        'tournament_size': 2,
        'mutation_rate': 0.05,

        # cgp parameters
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 3,
        'n_rows': 3,
        'levels_back': 2,
    }

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPConstantFloat])

    def objective(genome):

        torch.manual_seed(SEED)

        n_function_evaluations = 100

        graph = gp.CGPGraph(genome)
        f_graph = graph.compile_torch_class()

        def f_target(x):  # target function
            # return 2.7182 + x[0] - x[1]
            return 1. + x[:, 0] - x[:, 1]

        x = torch.Tensor(n_function_evaluations, params['n_inputs']).normal_()
        y = f_graph(x)

        loss = torch.mean((f_target(x) - y[:, 0]) ** 2)

        return -loss.item()

    # create population object that will be evolved
    pop = gp.CGPPopulation(
        params['n_parents'], params['n_offsprings'], params['n_breeding'], params['tournament_size'], params['mutation_rate'],
        params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)

    # generate initial parent population of size N
    pop.generate_random_parent_population()

    # generate initial offspring population of size N
    pop.generate_random_offspring_population()

    best_fitness = -1e15

    # perform evolution
    for i in range(params['generations']):

        # combine parent and offspring populations
        pop.create_combined_population()

        #  compute fitness for all objectives for all individuals
        pop.compute_fitness(objective)

        # sort population according to fitness & crowding distance
        pop.sort()

        # fill new parent population according to sorting
        pop.create_new_parent_population()

        # generate new offspring population from parent population
        pop.create_new_offspring_population()

        # perform local search to tune values of constants
        # TODO pop.local_search(objective)

        # for fixed inputs (seed is fixed in objective), fitness
        # should monotonously increase due to elitism
        assert best_fitness <= np.max(pop.fitness)
        best_fitness = np.max(pop.fitness)

    assert abs(np.mean(pop.fitness)) < 1e-10
