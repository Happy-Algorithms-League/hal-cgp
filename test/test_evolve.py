import numpy as np
import pytest
import torch
import time

import gp

SEED = np.random.randint(2 ** 31)


def objective_parallel_population(individual):

    if individual.fitness is not None:
        return individual

    torch.manual_seed(SEED)

    n_function_evaluations = 100

    graph = gp.CartesianGraph(individual.genome)
    f_graph = graph.to_torch()

    def f_target(x):  # target function
        return 1.0 + x[:, 0] - x[:, 1]

    x = torch.Tensor(n_function_evaluations, 2).normal_()
    y = f_graph(x)

    loss = torch.mean((f_target(x) - y[:, 0]) ** 2)

    individual.fitness = -loss.item()

    return individual


def _test_population(n_processes):

    population_params = {
        "n_parents": 5,
        "n_offsprings": 5,
        "max_generations": 2000,
        "n_breeding": 5,
        "tournament_size": 2,
        "mutation_rate": 0.05,
        "min_fitness": 1e-12,
    }

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": [gp.Add, gp.Sub, gp.Mul, gp.ConstantFloat],
    }

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    pop = gp.Population(
        population_params["n_parents"], population_params["mutation_rate"], SEED, genome_params
    )
    ea = gp.ea.MuPlusLambda(
        population_params["n_offsprings"],
        population_params["n_breeding"],
        population_params["tournament_size"],
        n_processes=n_processes,
    )

    gp.evolve(
        pop,
        objective_parallel_population,
        ea,
        population_params["max_generations"],
        population_params["min_fitness"],
    )

    assert pytest.approx(pop.champion.fitness == 0.0)

    return np.mean(pop.fitness_parents())


def test_parallel_population():

    time_per_n_processes = []
    fitness_per_n_processes = []
    for n_processes in [1, 2, 4]:
        t1 = time.time()
        fitness_per_n_processes.append(_test_population(n_processes))
        time_per_n_processes.append(time.time() - t1)
    assert fitness_per_n_processes[0] == pytest.approx(fitness_per_n_processes[1])
    assert fitness_per_n_processes[0] == pytest.approx(fitness_per_n_processes[2])


def test_pop_uses_own_rng():

    population_params = {
        "n_parents": 5,
        "n_offsprings": 5,
        "max_generations": 500,
        "n_breeding": 5,
        "tournament_size": 2,
        "mutation_rate": 0.05,
        "min_fitness": 1e-12,
    }

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": [gp.Add, gp.Sub, gp.Mul, gp.ConstantFloat],
    }

    pop = gp.Population(
        population_params["n_parents"], population_params["mutation_rate"], SEED, genome_params
    )

    # test for generating random parent population
    np.random.seed(SEED)

    pop._generate_random_parent_population()
    parents_0 = list(pop._parents)

    np.random.seed(SEED)

    pop._generate_random_parent_population()
    parents_1 = list(pop._parents)

    for p_0, p_1 in zip(parents_0, parents_1):
        assert p_0.genome.dna != p_1.genome.dna


def test_evolve_two_expressions():
    def _objective(individual):

        if individual.fitness is not None:
            return individual

        def f0(x):
            return x[0] * (x[0] + x[0])

        def f1(x):
            return (x[0] * x[1]) - x[1]

        y0 = gp.CartesianGraph(individual.genome[0]).to_func()
        y1 = gp.CartesianGraph(individual.genome[1]).to_func()

        loss = 0
        for _ in range(100):

            x0 = np.random.uniform(size=1)
            x1 = np.random.uniform(size=2)

            loss += (f0(x0) - y0(x0)) ** 2
            loss += (f1(x1) - y1(x1)) ** 2

        individual.fitness = -loss

        return individual

    population_params = {
        "n_parents": 5,
        "n_offsprings": 5,
        "max_generations": 1000,
        "n_breeding": 5,
        "tournament_size": 2,
        "mutation_rate": 0.05,
        "min_fitness": 1e-12,
    }

    genome_params = [
        {
            "n_inputs": 1,
            "n_outputs": 1,
            "n_columns": 3,
            "n_rows": 3,
            "levels_back": 2,
            "primitives": [gp.Add, gp.Mul],
        },
        {
            "n_inputs": 2,
            "n_outputs": 1,
            "n_columns": 2,
            "n_rows": 2,
            "levels_back": 2,
            "primitives": [gp.Sub, gp.Mul],
        },
    ]

    pop = gp.Population(
        population_params["n_parents"], population_params["mutation_rate"], SEED, genome_params
    )
    ea = gp.ea.MuPlusLambda(
        population_params["n_offsprings"],
        population_params["n_breeding"],
        population_params["tournament_size"],
    )

    gp.evolve(
        pop, _objective, ea, population_params["max_generations"], population_params["min_fitness"]
    )

    assert pytest.approx(pop.champion.fitness == 0.0)


def objective_speedup_parallel_evolve(individual):

    time.sleep(0.1)

    individual.fitness = np.random.rand()

    return individual


@pytest.mark.skip(reason="Test is not robust against execution in CI.")
def test_speedup_parallel_evolve():

    population_params = {
        "n_parents": 4,
        "n_offsprings": 4,
        "max_generations": 2,
        "n_breeding": 5,
        "tournament_size": 2,
        "mutation_rate": 0.05,
        "min_fitness": 1.0,
    }

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": [gp.Add, gp.Sub, gp.Mul, gp.ConstantFloat],
    }

    # Number of calls to objective: Number of parents + (Number of
    # parents + offspring) * (N_generations - 1) Initially, we need to
    # compute the fitness for all parents. Then we compute the fitness
    # for each parents and offspring in each iteration.
    n_calls_objective = population_params["n_parents"] + (
        population_params["n_parents"] + population_params["n_offsprings"]
    ) * (population_params["max_generations"] - 1)
    np.random.seed(SEED)

    # Serial execution

    for n_processes in [1, 2, 4]:
        pop = gp.Population(
            population_params["n_parents"], population_params["mutation_rate"], SEED, genome_params
        )
        ea = gp.ea.MuPlusLambda(
            population_params["n_offsprings"],
            population_params["n_breeding"],
            population_params["tournament_size"],
            n_processes=n_processes,
        )

        t0 = time.time()
        gp.evolve(
            pop,
            objective_speedup_parallel_evolve,
            ea,
            population_params["max_generations"],
            population_params["min_fitness"],
        )
        T = time.time() - t0
        if n_processes == 1:
            T_baseline = T
            # assert that total execution time is roughly equal to
            # number of objective calls x time per call
            assert T == pytest.approx(n_calls_objective * 0.1, rel=0.25)
        else:
            # assert that multiprocessing roughly follows a linear speedup.
            assert T == pytest.approx(T_baseline / n_processes, rel=0.25)