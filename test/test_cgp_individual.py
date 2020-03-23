import numpy as np
import pickle
import sys

sys.path.insert(0, "../")
import gp
from gp.cgp_individual import CGPIndividual


SEED = np.random.randint(2 ** 31)


def test_pickle_individual():

    primitives = [gp.CGPAdd]
    genome = gp.CGPGenome(1, 1, 1, 1, 1, primitives)
    individual = CGPIndividual(None, genome)

    with open("individual.pkl", "wb") as f:
        pickle.dump(individual, f)
