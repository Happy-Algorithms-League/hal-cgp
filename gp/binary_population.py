import numpy as np

from .abstract_population import AbstractPopulation
from .binary_individual import BinaryIndividual


class BinaryPopulation(AbstractPopulation):

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed,
                 genome_length, *, n_threads=1):
        super().__init__(n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed, n_threads=n_threads)

        self._genome_length = genome_length  # length of genome

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            individual = BinaryIndividual(fitness=None, genome=[0] * self._genome_length)
            individual.randomize_genome(self.rng)
            individuals.append(individual)

        return individuals

    def _crossover(self, breeding_pool):
        offsprings = []
        while len(offsprings) < self._n_offsprings:
            first_parent, second_parent = self.rng.permutation(breeding_pool)[:2]
            offsprings.append(first_parent.crossover(second_parent, self.rng))

        return offsprings

    def _mutate(self, offsprings):

        n_mutations = int(self._mutation_rate * len(offsprings[0].genome))
        assert n_mutations > 0

        for off in offsprings:
            off.mutate(n_mutations, self.rng)

        return offsprings
