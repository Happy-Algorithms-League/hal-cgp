import numpy as np

import gp


SEED = np.random.randint(2 ** 31)

population_params = {
    "n_parents": 5,
    "mutation_rate": 0.05,
    "seed": SEED,
}

genome_params = {
    "n_inputs": 2,
    "n_outputs": 1,
    "n_columns": 3,
    "n_rows": 3,
    "levels_back": 2,
    "primitives": [gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPConstantFloat],
}


def test_label():
    def objective_without_label(individual):
        assert True
        individual.fitness = -1
        return individual

    def objective_with_label(individual, label):
        assert label == "test"
        individual.fitness = -1
        return individual

    pop = gp.CGPPopulation(**population_params, genome_params=genome_params)

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

    pop = gp.CGPPopulation(**population_params, genome_params=genome_params)

    ea = gp.ea.MuPlusLambda(10, 10, 1)
    ea.initialize_fitness_parents(pop, objective)
    ea.step(pop, objective)
