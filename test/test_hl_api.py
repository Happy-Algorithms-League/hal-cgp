import functools
import numpy as np
import pytest
import time

import cgp


def _objective_test_population(individual, rng_seed):

    if individual.fitness is not None:
        return individual

    np.random.seed(rng_seed)

    n_function_evaluations = 100

    f_graph = individual.to_func()

    def f_target(x):  # target function
        return x[:, 0] - x[:, 1]

    x = np.random.normal(size=(n_function_evaluations, 2))
    y = np.empty(n_function_evaluations)
    for i, x_i in enumerate(x):
        y[i] = f_graph(x_i)[0]

    loss = np.mean((f_target(x) - y) ** 2)
    individual.fitness = -loss

    return individual


def _test_population(population_params, genome_params, ea_params):

    evolve_params = {"max_generations": 2000, "min_fitness": -1e-12}

    np.random.seed(population_params["seed"])

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)

    history = {}
    history["max_fitness_per_generation"] = []

    def recording_callback(pop):
        history["max_fitness_per_generation"].append(pop.champion.fitness)

    obj = functools.partial(_objective_test_population, rng_seed=population_params["seed"])
    cgp.evolve(pop, obj, ea, **evolve_params, callback=recording_callback)

    assert pop.champion.fitness >= evolve_params["min_fitness"]

    return history["max_fitness_per_generation"]


def test_parallel_population(population_params, genome_params, ea_params):
    """Test consistent evolution independent of the number of processes.
    """

    fitness_per_n_processes = []
    for n_processes in [1, 2, 4]:
        ea_params["n_processes"] = n_processes
        fitness_per_n_processes.append(
            _test_population(population_params, genome_params, ea_params)
        )
    assert fitness_per_n_processes[0] == pytest.approx(fitness_per_n_processes[1])
    assert fitness_per_n_processes[0] == pytest.approx(fitness_per_n_processes[2])


def test_evolve_two_expressions(population_params, ea_params):
    """Test evolution of multiple expressions simultaneously.
    """

    def _objective(individual):

        if individual.fitness is not None:
            return individual

        def f0(x):
            return x[0] * (x[0] + x[0])

        def f1(x):
            return (x[0] * x[1]) - x[1]

        y0 = cgp.CartesianGraph(individual.genome[0]).to_func()
        y1 = cgp.CartesianGraph(individual.genome[1]).to_func()

        loss = 0
        for _ in range(100):

            x0 = np.random.uniform(size=1)
            x1 = np.random.uniform(size=2)

            loss += float((f0(x0) - y0(x0)) ** 2)
            loss += float((f1(x1) - y1(x1)) ** 2)
        individual.fitness = -loss

        return individual

    # contains parameters for two distinct CartesianGraphs as list of
    # two dicts
    genome_params = [
        {
            "n_inputs": 1,
            "n_outputs": 1,
            "n_columns": 4,
            "n_rows": 2,
            "levels_back": 2,
            "primitives": (cgp.Add, cgp.Mul),
        },
        {
            "n_inputs": 2,
            "n_outputs": 1,
            "n_columns": 2,
            "n_rows": 2,
            "levels_back": 2,
            "primitives": (cgp.Sub, cgp.Mul),
        },
    ]

    evolve_params = {"max_generations": 2000, "min_fitness": -1e-12}

    np.random.seed(population_params["seed"])

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)

    cgp.evolve(pop, _objective, ea, **evolve_params)

    assert abs(pop.champion.fitness) == pytest.approx(0.0)


def _objective_speedup_parallel_evolve(individual):

    time.sleep(0.25)

    individual.fitness = np.random.rand()

    return individual


@pytest.mark.skip(reason="Test is not robust against execution in CI.")
def test_speedup_parallel_evolve(population_params, genome_params, ea_params):

    # use 4 parents and 4 offsprings to achieve even load on 2, 4
    # cores
    population_params["n_parents"] = 4
    ea_params["n_offsprings"] = 4

    evolve_params = {"max_generations": 5, "min_fitness": np.inf}

    # Number of calls to objective: Number of parents + (Number of
    # parents + offspring) * (N_generations - 1) Initially, we need to
    # compute the fitness for all parents. Then we compute the fitness
    # for each parents and offspring in each iteration.
    n_calls_objective = population_params["n_parents"] + (
        population_params["n_parents"] + ea_params["n_offsprings"]
    ) * (evolve_params["max_generations"] - 1)
    np.random.seed(population_params["seed"])

    time_per_objective_call = 0.25

    # Serial execution

    for n_processes in [1, 2, 4]:
        pop = cgp.Population(**population_params, genome_params=genome_params)

        ea = cgp.ea.MuPlusLambda(**ea_params, n_processes=n_processes)

        t0 = time.time()
        cgp.evolve(pop, _objective_speedup_parallel_evolve, ea, **evolve_params)
        T = time.time() - t0

        if n_processes == 1:
            T_baseline = T
            # assert that total execution time is roughly equal to
            # number of objective calls x time per call; serves as a
            # baseline for subsequent parallel evolutions
            assert T == pytest.approx(n_calls_objective * time_per_objective_call, rel=0.25)
        else:
            # assert that multiprocessing roughly follows a linear speedup.
            assert T == pytest.approx(T_baseline / n_processes, rel=0.25)
