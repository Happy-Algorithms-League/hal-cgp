import pickle

import gp
from gp.individual import Individual


def test_pickle_individual():

    primitives = [gp.Add]
    genome = gp.Genome(1, 1, 1, 1, 1, primitives)
    individual = Individual(None, genome)

    with open("individual.pkl", "wb") as f:
        pickle.dump(individual, f)
