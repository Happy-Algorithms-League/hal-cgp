import numpy as np
import pytest

import gp


def test_label(population_params, genome_params):
    def objective_without_label(individual):
        assert True
        individual.fitness = -1
        return individual

    def objective_with_label(individual, label):
        assert label == "test"
        individual.fitness = -1
        return individual

    pop = gp.Population(**population_params, genome_params=genome_params)

    ea = gp.ea.MuPlusLambda(1, 2, 1)
    ea.initialize_fitness_parents(pop, objective_without_label)
    ea.step(pop, objective_without_label)
    ea.step(pop, objective_with_label, label="test")


def test_fitness_contains_nan(population_params, genome_params):
    def objective(individual):
        if np.random.rand() < 0.5:
            individual.fitness = np.nan
        else:
            individual.fitness = np.random.rand()
        return individual

    pop = gp.Population(**population_params, genome_params=genome_params)

    ea = gp.ea.MuPlusLambda(10, 10, 1)
    ea.initialize_fitness_parents(pop, objective)
    ea.step(pop, objective)


def _objective_reevaluate_fitness(ind):
    raise RuntimeError()


def test_reevaluate_fitness(ea_params):

    ind = gp.individual.Individual(None, [])

    # first test reevaluate=False
    ea = gp.ea.MuPlusLambda(**ea_params, n_processes=2, reevaluate_fitness=False)

    # should NOT hit the exception if the fitness of the individual is
    # NOT None
    ind.fitness = 1.0
    ea._compute_fitness([ind], _objective_reevaluate_fitness)

    # should hit the exception if the fitness of the individual is
    # None
    ind.fitness = None
    with pytest.raises(RuntimeError):
        ea._compute_fitness([ind], _objective_reevaluate_fitness)

    # second test reevaluate=True
    ea = gp.ea.MuPlusLambda(**ea_params, n_processes=2, reevaluate_fitness=True)

    # should hit the exception if the fitness of the individual is
    # None
    ind.fitness = 1.0
    with pytest.raises(RuntimeError):
        ea._compute_fitness([ind], _objective_reevaluate_fitness)

    # should hit the exception if the fitness of the individual is
    # None
    ind.fitness = None
    with pytest.raises(RuntimeError):
        ea._compute_fitness([ind], _objective_reevaluate_fitness)
