import concurrent.futures
import functools
import multiprocessing as mp
import tempfile
import time

import numpy as np
import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_NON_CODING_GENE, ID_OUTPUT_NODE


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
def test_cache_decorator_produces_identical_history(
    individual_type, rng_seed, population_params, genome_params, ea_params
):
    pytest.importorskip("sympy")

    evolve_params = {"max_generations": 10, "min_fitness": 0.0}

    def f_target(x):
        return x[0] - x[1]

    def inner_objective(ind):
        expr = ind.to_sympy()
        np.random.seed(rng_seed)

        if individual_type == "SingleGenome":
            expr_unpacked = expr
        elif individual_type == "MultiGenome":
            expr_unpacked = expr[0]
        else:
            raise NotImplementedError

        loss = 0
        for x in np.random.uniform(size=(5, 2)):
            loss += (
                f_target(x) - float(expr_unpacked.subs({"x_0": x[0], "x_1": x[1]}).evalf())
            ) ** 2
        return loss

    @cgp.utils.disk_cache(
        tempfile.mkstemp()[1], compute_key=cgp.utils.compute_key_from_sympy_expr_and_args
    )
    def inner_objective_decorated(ind):
        return inner_objective(ind)

    def evolve(inner_objective):
        def objective(ind):
            ind.fitness = -inner_objective(ind)
            return ind

        if individual_type == "SingleGenome":
            pop = cgp.Population(**population_params, genome_params=genome_params)
        elif individual_type == "MultiGenome":
            pop = cgp.Population(**population_params, genome_params=[genome_params])
        else:
            raise NotImplementedError

        ea = cgp.ea.MuPlusLambda(**ea_params)

        history = {}
        history["fitness_champion"] = []

        def recording_callback(pop):
            history["fitness_champion"].append(pop.champion.fitness)

        cgp.evolve(pop, objective, ea, **evolve_params, callback=recording_callback)

        return history

    history = evolve(inner_objective)
    history_decorated = evolve(inner_objective_decorated)

    for fitness, fitness_decorated in zip(
        history["fitness_champion"], history_decorated["fitness_champion"]
    ):
        assert fitness == pytest.approx(fitness_decorated)


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
def test_fec_cache_decorator_produces_identical_history(
    individual_type, rng_seed, population_params, genome_params, ea_params
):

    evolve_params = {"max_generations": 10, "min_fitness": 0.0}

    def f_target(x):
        return x[:, 0] - x[:, 1]

    def inner_objective(ind):
        np.random.seed(rng_seed)

        if individual_type == "SingleGenome":
            f = ind.to_func()
        elif individual_type == "MultiGenome":
            f = ind.to_func()[0]
        else:
            raise NotImplementedError

        x = np.random.uniform(size=(5, 2))
        loss = np.sum((f(x[:, 0], x[:, 1]) - f_target(x)) ** 2)
        return loss

    @cgp.utils.disk_cache(
        tempfile.mkstemp()[1], compute_key=cgp.utils.compute_key_from_numpy_evaluation_and_args
    )
    def inner_objective_decorated(ind):
        return inner_objective(ind)

    def evolve(inner_objective):
        def objective(ind):
            ind.fitness = -inner_objective(ind)
            return ind

        if individual_type == "SingleGenome":
            pop = cgp.Population(**population_params, genome_params=genome_params)
        elif individual_type == "MultiGenome":
            pop = cgp.Population(**population_params, genome_params=[genome_params])
        else:
            raise NotImplementedError

        ea = cgp.ea.MuPlusLambda(**ea_params)

        history = {}
        history["fitness_champion"] = []

        def recording_callback(pop):
            history["fitness_champion"].append(pop.champion.fitness)

        cgp.evolve(pop, objective, ea, **evolve_params, callback=recording_callback)

        return history

    history = evolve(inner_objective)
    history_decorated = evolve(inner_objective_decorated)

    for fitness, fitness_decorated in zip(
        history["fitness_champion"], history_decorated["fitness_champion"]
    ):
        assert fitness == pytest.approx(fitness_decorated)


@cgp.utils.disk_cache(
    tempfile.mkstemp()[1], compute_key=cgp.utils.compute_key_from_numpy_evaluation_and_args
)
def _fec_cache_decorator_with_multiple_inputs_multiple_outputs_objective(ind):
    f = ind.to_numpy()
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    y = f(x[:, 0], x[:, 1])
    return y


def test_fec_cache_decorator_with_multiple_inputs_multiple_outputs(genome_params):

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 2,
        "n_columns": 1,
        "n_rows": 1,
        "levels_back": None,
        "primitives": (cgp.Add, cgp.Sub),
    }

    genome0 = cgp.Genome(**genome_params)
    # [f0(x), f1(x)] = [x_0, x_0 + x_1]
    genome0.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        0,
        ID_NON_CODING_GENE,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    ind0 = cgp.IndividualSingleGenome(genome0)

    genome1 = cgp.Genome(**genome_params)
    # [f0(x), f1(x)] = [x_0, x_0 - x_1]
    genome1.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        1,
        0,
        1,
        ID_OUTPUT_NODE,
        0,
        ID_NON_CODING_GENE,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    ind1 = cgp.IndividualSingleGenome(genome1)

    y0 = _fec_cache_decorator_with_multiple_inputs_multiple_outputs_objective(ind0)

    # next call should *not* use the cached value despite the first
    # dimension being identical
    y1 = _fec_cache_decorator_with_multiple_inputs_multiple_outputs_objective(ind1)

    assert y0[0] == pytest.approx(y1[0])
    assert y0[1] != pytest.approx(y1[1])


@cgp.utils.disk_cache(tempfile.mkstemp()[1])
def _cache_decorator_objective_single_process(ind, sleep_time):
    time.sleep(sleep_time)  # simulate long execution time
    return 0.0


@cgp.utils.disk_cache(tempfile.mkstemp()[1], file_lock=mp.Lock())
def _cache_decorator_objective_two_processes(ind, sleep_time):
    time.sleep(sleep_time)  # simulate long execution time
    return 0.0


@pytest.mark.parametrize("n_processes", [1, 2])
def test_cache_decorator(n_processes, individual):
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
    x = [individual, individual]

    # WARNING: below the number of processes is *not* taken into
    # account in the timing; one would expect a two-fold speedup when
    # called with n_processes = 2, however, TravisCI seems to limit
    # the number of parallel processes; accounting for the number of
    # processes will hence make this test fail in continous
    # integration; the speedup of the second call with respect to the
    # first can nevertheless be tested

    # first call should take long due to sleep; at least 90% of the
    # sleep time; to account for possible timing measurement
    # inaccuracies we do not choose 100%
    t0 = time.time()
    evaluate_objective_on_list(x)
    assert (time.time() - t0) > (0.9 * sleep_time)

    # second call should be faster as result is retrieved from cache;
    # at most 40% of the sleep time; to account for possible timing
    # measurement inaccuracies and process spin up/down time in
    # TravisCI we should not choose less
    t0 = time.time()
    evaluate_objective_on_list(x)
    assert (time.time() - t0) < (0.4 * sleep_time)


def test_cache_decorator_consistency(individual):

    cache_fn = tempfile.mkstemp()[1]
    x = 2

    @cgp.utils.disk_cache(cache_fn)
    def objective_f(ind):
        return x

    # call objective_f once to initialize the cache
    assert objective_f(individual) == pytest.approx(x)

    # decorating a different function with different output using same
    # filename should raise an error
    with pytest.raises(RuntimeError):

        @cgp.utils.disk_cache(cache_fn)
        def objective_g(ind):
            return x ** 2

    # decorating a different function with identical output using the
    # same filename should NOT raise an error
    @cgp.utils.disk_cache(cache_fn)
    def objective_h(ind):
        return x


def test_cache_decorator_does_not_compare_infinite_return_values(individual):
    cache_fn = tempfile.mkstemp()[1]

    @cgp.utils.disk_cache(cache_fn)
    def objective_f(ind, x):
        try:
            return 1.0 / x
        except ZeroDivisionError:
            return np.inf

    # first call produces infinite return value, identical to
    # objective_g although in general their return values are
    # different
    objective_f(individual, 0.0)
    # second call produces a finite return value which should be used
    # to check consistency
    objective_f(individual, 2.0)

    # since the consistency check uses the finite return value it
    # should detect that the two objectives are indeed different
    with pytest.raises(RuntimeError):

        @cgp.utils.disk_cache(cache_fn)
        def objective_g(ind, x):
            try:
                return 2.0 / x
            except ZeroDivisionError:
                return np.inf


def test_cache_decorator_does_nothing_for_nonexistent_file():
    @cgp.utils.disk_cache("nonexistent_file.pkl")
    def objective(ind, x):
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


def test_fec_cache_decorator_with_additional_arguments(genome_params, rng, rng_seed):
    def f_target(x):
        return x[:, 0] - x[:, 1]

    @cgp.utils.disk_cache(
        tempfile.mkstemp()[1], compute_key=cgp.utils.compute_key_from_numpy_evaluation_and_args
    )
    def inner_objective(ind, n_samples):
        np.random.seed(rng_seed)

        f = ind.to_func()

        x = np.random.uniform(size=(n_samples, 2))
        loss = np.sum((f(x[:, 0], x[:, 1]) - f_target(x)) ** 2)
        return loss

    g = cgp.Genome(**genome_params)
    g.randomize(rng)
    ind = cgp.IndividualSingleGenome(g)

    # call with 5 and 10 samples, results should be different since
    # the second call should *not* return the cached value
    y0 = inner_objective(ind, 5)
    y1 = inner_objective(ind, 10)

    assert y0 != pytest.approx(y1)


def test_custom_compute_key_for_disk_cache(individual, rng):
    @cgp.utils.disk_cache(
        tempfile.mkstemp()[1], compute_key=cgp.utils.compute_key_from_numpy_evaluation_and_args
    )
    def inner_objective(ind):
        return ind.to_func()(1.0, 2.0)

    def my_compute_key(ind):
        return 0

    @cgp.utils.disk_cache(tempfile.mkstemp()[1], compute_key=my_compute_key)
    def inner_objective_custom_compute_key(ind):
        return ind.to_func()(1.0, 2.0)

    individual0 = individual.clone()
    individual0.genome.randomize(rng)
    individual1 = individual.clone()
    individual1.genome.randomize(rng)

    loss0 = inner_objective(individual0)
    loss1 = inner_objective(individual1)

    assert loss0 != pytest.approx(loss1)

    loss0 = inner_objective_custom_compute_key(individual0)
    loss1 = inner_objective_custom_compute_key(individual1)

    assert loss0 == pytest.approx(loss1)
