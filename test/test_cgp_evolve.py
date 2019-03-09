import numpy as np
import sys
import torch
import time

sys.path.insert(0, '../')
import gp

SEED = np.random.randint(2 ** 31)


def objective(individual):

    if individual.fitness is not None:
        return individual

    torch.manual_seed(SEED)

    n_function_evaluations = 100

    graph = gp.CGPGraph(individual.genome)
    f_graph = graph.compile_torch_class()

    def f_target(x):  # target function
        # return 2.7182 + x[0] - x[1]
        return 1. + x[:, 0] - x[:, 1]

    x = torch.Tensor(n_function_evaluations, 2).normal_()
    y = f_graph(x)

    loss = torch.mean((f_target(x) - y[:, 0]) ** 2)

    individual.fitness = -loss.item()

    return individual


def _test_cgp_population(n_threads):

    params = {
        # evo parameters
        'n_parents': 5,
        'n_offsprings': 5,
        'max_generations': 500,
        'n_breeding': 5,
        'tournament_size': 2,
        'mutation_rate': 0.05,
        'min_fitness': 1e-12,

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

    pop = gp.CGPPopulation(
        params['n_parents'], params['n_offsprings'], params['n_breeding'], params['tournament_size'], params['mutation_rate'], SEED, params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives, n_threads=n_threads)

    history_fitness, _ = gp.evolve(pop, objective, params['max_generations'], params['min_fitness'])

    assert abs(np.mean(pop.fitness_parents())) < 1e-10

    return np.mean(pop.fitness_parents())


def test_parallel_cgp_population():

    time_per_n_threads = []
    fitness_per_n_threads = []
    for n_threads in [1, 2, 4]:
        t1 = time.time()
        fitness_per_n_threads.append(_test_cgp_population(n_threads))
        time_per_n_threads.append(time.time() - t1)

    assert abs(fitness_per_n_threads[0] - fitness_per_n_threads[1]) < 1e-10
    assert abs(fitness_per_n_threads[0] - fitness_per_n_threads[2]) < 1e-10


def test_cgp_pop_uses_own_rng():

    params = {
        # evo parameters
        'n_parents': 5,
        'n_offsprings': 5,
        'max_generations': 500,
        'n_breeding': 5,
        'tournament_size': 2,
        'mutation_rate': 0.05,
        'min_fitness': 1e-12,

        # cgp parameters
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 3,
        'n_rows': 3,
        'levels_back': 2,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPConstantFloat])

    pop = gp.CGPPopulation(
        params['n_parents'], params['n_offsprings'], params['n_breeding'], params['tournament_size'], params['mutation_rate'], SEED, params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)

    # test for generating random parent population
    np.random.seed(SEED)

    pop.generate_random_parent_population()
    parents_0 = list(pop._parents)

    np.random.seed(SEED)

    pop.generate_random_parent_population()
    parents_1 = list(pop._parents)

    for p_0, p_1 in zip(parents_0, parents_1):
        assert p_0.genome.dna != p_1.genome.dna

    # test for generating random offspring population
    np.random.seed(SEED)

    pop.generate_random_offspring_population()
    offsprings_0 = list(pop._offsprings)

    np.random.seed(SEED)

    pop.generate_random_offspring_population()
    offsprings_1 = list(pop._offsprings)

    for o_0, o_1 in zip(offsprings_0, offsprings_1):
        assert o_0.genome.dna != o_1.genome.dna

    # test for generating new offspring population
    for i, p in enumerate(pop._parents):  # dummy fitness
        p.fitness = i
    np.random.seed(SEED)

    pop.create_new_offspring_population()
    offsprings_0 = list(pop._offsprings)

    np.random.seed(SEED)

    pop.create_new_offspring_population()
    offsprings_1 = list(pop._offsprings)

    for o_0, o_1 in zip(offsprings_0, offsprings_1):
        assert o_0.genome.dna != o_1.genome.dna
