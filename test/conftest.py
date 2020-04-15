import gp
import numpy as np
from pytest import fixture


@fixture
def rng_seed():
    return np.random.randint(2 ** 31)


@fixture
def genome_params():
    return {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": [gp.Add, gp.Sub, gp.ConstantFloat],
    }


@fixture
def population_params(mutation_rate, rng_seed):
    return {"n_parents": 5, "mutation_rate": mutation_rate, "seed": rng_seed}


@fixture
def mutation_rate():
    return np.random.rand()
