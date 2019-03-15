import numpy as np

from .abstract_population import AbstractPopulation
from .binary_individual import BinaryIndividual


class BinaryPopulation(AbstractPopulation):

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed,
                 genome_params, *, n_threads=1):
        super().__init__(n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed, n_threads=n_threads)

        self._genome_params = genome_params

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            individual = BinaryIndividual(fitness=None, genome=None)
            individual.randomize_genome(self._genome_params, self.rng)
            individuals.append(individual)

        return individuals
