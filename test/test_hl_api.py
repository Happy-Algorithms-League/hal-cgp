import functools
import time

import numpy as np
import pytest

import cgp


def _objective_test_population(individual, rng_seed):

    if not individual.fitness_is_None():
        return individual

    n_function_evaluations = 100

    f_graph = individual.to_func()

    def f_target(x):  # target function
        return x[:, 0] - x[:, 1]

    rng = np.random.RandomState(seed=rng_seed)
    x = rng.normal(size=(n_function_evaluations, 2))
    y = f_graph(x[:, 0], x[:, 1])
    # y = np.empty(n_function_evaluations)
    # for i, x_i in enumerate(x):
    #     y[i] = f_graph(x_i)

    loss = np.mean((f_target(x) - y) ** 2)
    individual.fitness = -loss

    return individual


def _test_population(population_params, genome_params, ea_params):

    evolve_params = {"max_generations": 2000, "termination_fitness": -1e-12}

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)

    history = {}
    history["max_fitness_per_generation"] = []

    def recording_callback(pop):
        history["max_fitness_per_generation"].append(pop.champion.fitness)

    obj = functools.partial(_objective_test_population, rng_seed=population_params["seed"])
    cgp.evolve(obj, pop, ea, **evolve_params, callback=recording_callback)

    assert pop.champion.fitness >= evolve_params["termination_fitness"]

    return history["max_fitness_per_generation"]


def test_parallel_population(population_params, genome_params, ea_params, rng_seed):
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


def test_evolve_two_expressions(population_params, ea_params, rng_seed):
    """Test evolution of multiple expressions simultaneously.
    """

    def _objective(individual):

        if not individual.fitness_is_None():
            return individual

        rng = np.random.RandomState(rng_seed)

        def f0(x):
            return x[0] * (x[0] + x[0])

        def f1(x):
            return (x[0] * x[1]) - x[1]

        y0 = cgp.CartesianGraph(individual.genome[0]).to_func()
        y1 = cgp.CartesianGraph(individual.genome[1]).to_func()

        loss = 0
        for _ in range(100):

            x0 = rng.uniform(size=1)
            x1 = rng.uniform(size=2)

            loss += float((f0(x0) - y0(x0)) ** 2)
            loss += float((f1(x1) - y1(x1[0], x1[1])) ** 2)
        individual.fitness = -loss

        return individual

    # contains parameters for two distinct CartesianGraphs as list of
    # two dicts
    genome_params = [
        {"n_inputs": 1, "n_outputs": 1, "n_hidden_units": 4, "primitives": (cgp.Add, cgp.Mul),},
        {"n_inputs": 2, "n_outputs": 1, "n_hidden_units": 4, "primitives": (cgp.Sub, cgp.Mul),},
    ]

    evolve_params = {"max_generations": 2000, "termination_fitness": -1e-12}

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)

    cgp.evolve(_objective, pop, ea, **evolve_params)

    assert abs(pop.champion.fitness) == pytest.approx(0.0)


def _objective_speedup_parallel_evolve(individual, rng_seed):

    time.sleep(0.25)

    rng = np.random.RandomState(rng_seed)
    individual.fitness = rng.rand()

    return individual


@pytest.mark.skip(reason="Test is not robust against execution in CI.")
def test_speedup_parallel_evolve(population_params, genome_params, ea_params, rng_seed):

    # use 4 parents and 4 offsprings to achieve even load on 2, 4
    # cores
    population_params["n_parents"] = 4
    ea_params["n_offsprings"] = 4

    evolve_params = {"max_generations": 5, "termination_fitness": np.inf}

    # Number of calls to objective: Number of parents + (Number of
    # parents + offspring) * (N_generations - 1) Initially, we need to
    # compute the fitness for all parents. Then we compute the fitness
    # for each parents and offspring in each iteration.
    n_calls_objective = population_params["n_parents"] + (
        population_params["n_parents"] + ea_params["n_offsprings"]
    ) * (evolve_params["max_generations"] - 1)

    time_per_objective_call = 0.25

    obj = functools.partial(_objective_speedup_parallel_evolve, rng_seed=rng_seed)
    for n_processes in [1, 2, 4]:
        pop = cgp.Population(**population_params, genome_params=genome_params)

        ea = cgp.ea.MuPlusLambda(**ea_params, n_processes=n_processes)

        t0 = time.time()
        cgp.evolve(obj, pop, ea, **evolve_params)
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


def test_non_persistent_default_population(rng):
    def objective(ind):
        ind.fitness = rng.rand()
        return ind

    pop0 = cgp.evolve(objective, max_generations=5)
    pop1 = cgp.evolve(objective, max_generations=5)

    # since these were two independent runs with random fitness assignments we
    # would expect the champions to be different
    assert pop0.champion.fitness != pytest.approx(pop1.champion.fitness)
