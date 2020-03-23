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
        "primitives": [gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPConstantFloat],
    }

    pop = gp.CGPPopulation(
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

    primitives_str = ["CGPAdd", "CGPSub", "CGPMul"]
    primitives = gp.utils.primitives_from_class_names(primitives_str)
    assert issubclass(primitives[0], gp.CGPAdd)
    assert issubclass(primitives[1], gp.CGPSub)
    assert issubclass(primitives[2], gp.CGPMul)

    # make sure custom classes are registered as well
    class MyCustomCGPNodeClass(gp.cgp_node.CGPNode):
        pass

    primitives = gp.utils.primitives_from_class_names(["MyCustomCGPNodeClass"])
    assert issubclass(primitives[0], MyCustomCGPNodeClass)
