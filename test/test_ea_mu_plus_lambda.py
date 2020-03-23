import numpy as np
import sys

sys.path.insert(0, "../")
import gp


SEED = np.random.randint(2 ** 31)


def test_label():
    def objective_without_label(individual):
        assert True
        individual.fitness = -1
        return individual

    def objective_with_label(individual, label):
        assert label == "test"
        individual.fitness = -1
        return individual

    pop = gp.BinaryPopulation(1, 0.5, SEED, {"genome_length": 2, "primitives": [0, 1]})
    ea = gp.ea.MuPlusLambda(1, 2, 1)
    ea.initialize_fitness_parents(pop, objective_without_label)
    ea.step(pop, objective_without_label)
    ea.step(pop, objective_with_label, label="test")


def test_fitness_contains_nan():
    def objective(individual):
        if np.random.rand() < 0.5:
            individual.fitness = np.nan
        else:
            individual.fitness = np.random.rand()
        return individual

    pop = gp.BinaryPopulation(5, 0.5, SEED, {"genome_length": 2, "primitives": [0, 1]})
    ea = gp.ea.MuPlusLambda(10, 10, 1)
    ea.initialize_fitness_parents(pop, objective)
    ea.step(pop, objective)
