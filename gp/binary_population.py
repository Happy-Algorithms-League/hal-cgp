import numpy as np

from .abstract_population import AbstractPopulation
from .abstract_individual import AbstractIndividual


class BinaryIndividual(AbstractIndividual):

    def __init__(self, fitness, genome):
        super().__init__(fitness, genome)

    def clone(self):
        return BinaryIndividual(self.fitness, self.genome)


class BinaryPopulation(AbstractPopulation):

    def __init__(self, n_parents, n_offsprings, genome_length, n_breeding, tournament_size, mutation_rate, seed, *, n_threads=1):
        super().__init__(n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed, n_threads=n_threads)

        self._genome_length = genome_length  # length of genome

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            individuals.append(
                BinaryIndividual(None, str(self.rng.randint(10 ** self._genome_length)).zfill(self._genome_length)))
        return individuals

    def _crossover(self, breeding_pool):
        offsprings = []
        while len(offsprings) < self._n_offsprings:
            # choose parents and perform crossover at random position in genome
            parents = self.rng.permutation(breeding_pool)[:2]
            split_pos = self.rng.randint(self._genome_length)
            offsprings.append(
                BinaryIndividual(None, parents[0].genome[:split_pos] + parents[1].genome[split_pos:]))

        return offsprings

    def _mutate(self, offsprings):

        n_mutations = int(self._mutation_rate * len(offsprings[0].genome))
        assert n_mutations > 0

        for off in offsprings:
            for i in range(n_mutations):
                # mutate random gene
                genome = list(off.genome)
                genome[self.rng.randint(self._genome_length)] = str(self.rng.randint(10))
                off.genome = ''.join(genome)

        return offsprings
