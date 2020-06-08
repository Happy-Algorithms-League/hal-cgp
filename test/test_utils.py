import concurrent.futures
import functools
import numpy as np
import pytest
import tempfile
import time

import cgp


@cgp.utils.disk_cache(tempfile.mkstemp()[1])
def _cache_decorator_objective_single_process(s, sleep_time):
    time.sleep(sleep_time)  # simulate long execution time
    return s


@cgp.utils.disk_cache(tempfile.mkstemp()[1])
def _cache_decorator_objective_two_processes(s, sleep_time):
    time.sleep(sleep_time)  # simulate long execution time
    return s


@pytest.mark.parametrize("n_processes", [1, 2])
def test_cache_decorator(n_processes):
    def evaluate_objective_on_list(x):
        if n_processes == 1:
            objective = functools.partial(
                _cache_decorator_objective_single_process, sleep_time=sleep_time
            )
            return list(map(objective, x))
        else:
            objective = functools.partial(
                _cache_decorator_objective_two_processes, sleep_time=sleep_time
            )
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_processes) as executor:
                return list(executor.map(objective, x))

    sleep_time = 1.0
    x = ["test0", "test1"]

    # WARNING: below the number of processes is *not* taken into
    # account in the timing; one would expect a two-fold speedup when
    # called with n_processes = 2, however, TravisCI seems to limit
    # the number of parallel processes; accounting for the number of
    # processes will hence make this test fail in continous
    # integration; the speedup of the second call with respect to the
    # first can nevertheless be tested

    # first call should take long due to sleep
    t0 = time.time()
    evaluate_objective_on_list(x)
    assert (time.time() - t0) > (sleep_time / 2.0)

    # second call should be faster as result is retrieved from cache
    t0 = time.time()
    evaluate_objective_on_list(x)
    assert (time.time() - t0) < (sleep_time / 5.0)


def test_cache_decorator_consistency():

    cache_fn = tempfile.mkstemp()[1]
    x = 2

    @cgp.utils.disk_cache(cache_fn)
    def objective_f(x):
        return x

    # call objective_f once to initialize the cache
    assert objective_f(x) == pytest.approx(x)

    # decorating a different function with different output using same
    # filename should raise an error
    with pytest.raises(RuntimeError):

        @cgp.utils.disk_cache(cache_fn)
        def objective_g(x):
            return x ** 2

    # decorating a different function with identical output using the
    # same filename should NOT raise an error
    @cgp.utils.disk_cache(cache_fn)
    def objective_h(x):
        return x


def objective_history_recording(individual):
    individual.fitness = 1.0
    return individual


def test_history_recording(population_params, genome_params, ea_params):

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    evolve_params = {"max_generations": 2, "min_fitness": 1.0}

    history = {}
    history["fitness"] = np.empty(
        (evolve_params["max_generations"], population_params["n_parents"])
    )
    history["fitness_champion"] = np.empty(evolve_params["max_generations"])
    history["champion"] = []

    def recording_callback(pop):
        history["fitness"][pop.generation] = pop.fitness_parents()
        history["fitness_champion"][pop.generation] = pop.champion.fitness
        history["champion"].append(pop.champion)

    cgp.evolve(pop, objective_history_recording, ea, **evolve_params, callback=recording_callback)

    assert np.all(history["fitness"] == pytest.approx(1.0))
    assert np.all(history["fitness_champion"] == pytest.approx(1.0))
    assert "champion" in history


def test_primitives_from_class_names():

    primitives_str = ["Add", "Sub", "Mul"]
    primitives = cgp.utils.primitives_from_class_names(primitives_str)
    assert issubclass(primitives[0], cgp.Add)
    assert issubclass(primitives[1], cgp.Sub)
    assert issubclass(primitives[2], cgp.Mul)

    # make sure custom classes are registered as well
    class MyCustomNodeClass(cgp.node.Node):
        pass

    primitives = cgp.utils.primitives_from_class_names(["MyCustomNodeClass"])
    assert issubclass(primitives[0], MyCustomNodeClass)


def test_primitives_from_class_names_for_genome(genome_params):
    primitives_str = ("Add", "Sub", "Mul")
    primitives = cgp.utils.primitives_from_class_names(primitives_str)

    genome_params["primitives"] = primitives

    cgp.Genome(**genome_params)
