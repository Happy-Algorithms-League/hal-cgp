import functools
import numpy as np
import pytest

import cgp


def test_objective_with_label(population_params, genome_params):
    def objective_without_label(individual):
        individual.fitness = -2.0
        return individual

    def objective_with_label(individual, label):
        assert label == "test"
        individual.fitness = -1.0
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(1, 2, 1)
    ea.initialize_fitness_parents(pop, objective_without_label)

    ea.step(pop, objective_without_label)
    assert pop.champion.fitness == pytest.approx(-2.0)

    obj = functools.partial(objective_with_label, label="test")
    ea.step(pop, obj)
    assert pop.champion.fitness == pytest.approx(-1.0)


def test_fitness_contains_nan(population_params, genome_params):
    def objective(individual):
        if np.random.rand() < 0.5:
            individual.fitness = np.nan
        else:
            individual.fitness = np.random.rand()
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(10, 10, 1)
    ea.initialize_fitness_parents(pop, objective)
    ea.step(pop, objective)
