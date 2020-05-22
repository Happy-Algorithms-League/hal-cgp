import gp
from pytest import fixture


@fixture
def rng_seed():
    return 1234


@fixture
def genome_params():
    return {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": (gp.Add, gp.Sub, gp.ConstantFloat),
    }


@fixture
def population_params(mutation_rate, rng_seed):
    return {"n_parents": 5, "mutation_rate": mutation_rate, "seed": rng_seed}


@fixture
def ea_params():
    return {"n_offsprings": 5, "n_breeding": 5, "tournament_size": 2}


@fixture
def mutation_rate():
    return 0.05


@fixture
def population_simple_fitness(population_params, genome_params):
    pop = gp.Population(**population_params, genome_params=genome_params)

    for i, parent in enumerate(pop.parents):
        parent.fitness = i

    return pop
