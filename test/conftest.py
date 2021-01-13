import numpy as np
from pytest import fixture

import cgp


@fixture
def rng_seed():
    return 1234


@fixture
def rng(rng_seed):
    return np.random.RandomState(rng_seed)


@fixture
def genome_params():
    return {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": (cgp.Add, cgp.Sub, cgp.ConstantFloat),
    }


@fixture
def genome_params_list(genome_params):
    return [genome_params]


@fixture
def population_params(rng_seed):
    return {"n_parents": 5, "seed": rng_seed}


@fixture
def ea_params(n_offsprings, tournament_size, mutation_rate):
    return {
        "n_offsprings": n_offsprings,
        "tournament_size": tournament_size,
        "mutation_rate": mutation_rate,
    }


@fixture
def mutation_rate():
    return 0.05


@fixture
def population_simple_fitness(population_params, genome_params):
    pop = cgp.Population(**population_params, genome_params=genome_params)

    for i, parent in enumerate(pop.parents):
        parent.fitness = float(i)

    return pop


@fixture
def local_search_params():
    return {"lr": 1e-3, "gradient_steps": 9}


@fixture
def n_offsprings():
    return 5


@fixture
def tournament_size():
    return 2


@fixture
def individual(genome_params, rng):
    g = cgp.Genome(**genome_params)
    g.randomize(rng)
    return cgp.IndividualSingleGenome(g)
