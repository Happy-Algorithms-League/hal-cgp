import numpy as np
import pytest
import tempfile
import time

import gp


SEED = np.random.randint(2 ** 31)


def test_cache_decorator():

    sleep_time = 0.1

    @gp.utils.disk_cache(tempfile.mkstemp()[1])
    def objective(label):
        time.sleep(sleep_time)  # simulate long execution time
        return label

    # first call should take long due to sleep
    t0 = time.time()
    objective("test")
    assert time.time() - t0 > sleep_time / 2.0

    # second call should be faster as result is retrieved from cache
    t0 = time.time()
    objective("test")
    assert time.time() - t0 < sleep_time / 2.0


def test_cache_decorator_consistency():

    cache_fn = tempfile.mkstemp()[1]

    @gp.utils.disk_cache(cache_fn)
    def objective_f(x):
        return x

    @gp.utils.disk_cache(cache_fn)
    def objective_g(x):
        return x ** 2

    @gp.utils.disk_cache(cache_fn)
    def objective_h(x):
        return x

    @gp.utils.disk_cache(cache_fn)
    def objective_f_2d(x, y):
        return x

    @gp.utils.disk_cache(cache_fn)
    def objective_f_with_kwargs(x, test=None):
        return x

    x = 2

    assert objective_f(x) == pytest.approx(x)
    assert objective_h(x) == pytest.approx(x)

    # calling a different function decorated with same filename but
    # different output should raise an error
    with pytest.raises(RuntimeError):
        objective_g(x)

    # calling a different function decorated with same filename and
    # same output should not raise an error
    objective_h(x)

    # calling a different function decorated with same filename and
    # same output but different arguments should raise an error
    with pytest.raises(RuntimeError):
        objective_f_2d(x, x)

    # calling a different function decorated with same filename and
    # same output but different keyword arguments should raise an
    # error
    with pytest.raises(RuntimeError):
        objective_f_with_kwargs(x, test=2)


def objective_history_recording(individual):
    individual.fitness = 1.0
    return individual


def test_history_recording():

    population_params = {
        "n_parents": 4,
        "n_offsprings": 4,
        "max_generations": 2,
        "n_breeding": 5,
        "tournament_size": 2,
        "mutation_rate": 0.05,
        "min_fitness": 2.0,
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
    ea = gp.ea.MuPlusLambda(
        population_params["n_offsprings"],
        population_params["n_breeding"],
        population_params["tournament_size"],
    )

    history = {}
    history["fitness"] = np.empty(
        (population_params["max_generations"], population_params["n_parents"])
    )
    history["fitness_champion"] = np.empty(population_params["max_generations"])
    history["expr_champion"] = []

    def recording_callback(pop):
        history["fitness"][pop.generation] = pop.fitness_parents()
        history["fitness_champion"][pop.generation] = pop.champion.fitness
        history["expr_champion"].append(str(pop.champion.to_sympy(simplify=True)[0]))

    gp.evolve(
        pop,
        objective_history_recording,
        ea,
        population_params["max_generations"],
        population_params["min_fitness"],
        callback=recording_callback,
    )

    assert np.all(history["fitness"] == pytest.approx(1.0))
    assert np.all(history["fitness_champion"] == pytest.approx(1.0))
    assert "expr_champion" in history


def test_primitives_from_class_names():

    primitives_str = ["Add", "Sub", "Mul"]
    primitives = gp.utils.primitives_from_class_names(primitives_str)
    assert issubclass(primitives[0], gp.Add)
    assert issubclass(primitives[1], gp.Sub)
    assert issubclass(primitives[2], gp.Mul)

    # make sure custom classes are registered as well
    class MyCustomNodeClass(gp.node.Node):
        pass

    primitives = gp.utils.primitives_from_class_names(["MyCustomNodeClass"])
    assert issubclass(primitives[0], MyCustomNodeClass)
