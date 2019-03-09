import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys

sys.path.insert(0, '../')
import gp
from gp.abstract_population import AbstractPopulation

SEED = np.random.randint(2 ** 31)


def test_mutation_rate_within_bounds():
    mutation_rate = 0.
    with pytest.raises(ValueError):
        pop = AbstractPopulation(1, 1, 1, 1, mutation_rate, 1)

    mutation_rate = 1.
    with pytest.raises(ValueError):
        pop = AbstractPopulation(1, 1, 1, 1, mutation_rate, 1)

    mutation_rate = 0.5
    pop = AbstractPopulation(1, 1, 1, 1, mutation_rate, 1)


def test_label():

    def objective_without_label(individual):
        assert True

    def objective_with_label(individual, label):
        assert label == 'test'

    pop = AbstractPopulation(1, 1, 1, 1, 0.5, 1)
    pop._combined = [1]
    pop.compute_fitness(objective_without_label)
    pop.compute_fitness(objective_with_label, label='test')
