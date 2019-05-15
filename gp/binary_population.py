import numpy as np

from .abstract_population import AbstractPopulation
from .binary_individual import BinaryIndividual, BinaryGenome


class BinaryPopulation(AbstractPopulation):

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed,
                 genome_params, *, n_threads=1):
        super().__init__(n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed, n_threads=n_threads)

        self._genome_params = genome_params

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            genome = BinaryGenome(self._genome_params['genome_length'], self._genome_params['primitives'])
            individual = BinaryIndividual(fitness=None, genome=genome)
            individual.randomize_genome(self.rng)
            individuals.append(individual)

        return individuals
