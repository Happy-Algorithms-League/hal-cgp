import numpy as np
import matplotlib.pyplot as plt
import pytest
import sys

sys.path.insert(0, '../')
import gp


SEED = np.random.randint(2 ** 31)


def test_label():

    def objective_without_label(individual):
        assert True
        individual.fitness = -1
        return individual

    def objective_with_label(individual, label):
        assert label == 'test'
        individual.fitness = -1
        return individual

    pop = gp.BinaryPopulation(1, 0.5, SEED, {'genome_length': 2, 'primitives': [0, 1]})
    ea = gp.ea.MuPlusLambda(1, 1, 2, 1)
    ea.initialize_fitness_parents(pop, objective_without_label)
    ea.step(pop, objective_without_label)
    ea.step(pop, objective_with_label, label='test')
