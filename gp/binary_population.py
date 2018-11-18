import numpy as np

from .abstract_population import AbstractPopulation, Individual


class BinaryPopulation(AbstractPopulation):

    def __init__(self, n_parents, n_offsprings, genome_length, n_breeding, tournament_size, mutation_rate):
        super().__init__(n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate)

        self._genome_length = genome_length  # length of genome

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            individuals.append(
                Individual(None, str(np.random.randint(10 ** self._genome_length)).zfill(self._genome_length)))
        return individuals

    def _crossover(self, breeding_pool):
        offsprings = []
        while len(offsprings) < self._n_offsprings:
            # choose parents and perform crossover at random position in genome
            parents = np.random.permutation(breeding_pool)[:2]
            split_pos = np.random.randint(self._genome_length)
            offsprings.append(
                Individual(None, parents[0].genome[:split_pos] + parents[1].genome[split_pos:]))

        return offsprings

    def _mutate(self, offsprings):

        n_mutations = int(self._mutation_rate * len(offsprings[0].genome))
        assert n_mutations > 0

        for off in offsprings:
            for i in range(n_mutations):
                # mutate random gene
                genome = list(off.genome)
                genome[np.random.randint(self._genome_length)] = str(np.random.randint(10))
                off.genome = ''.join(genome)

        return offsprings

    def _clone_individual(self, ind):
        return Individual(ind.fitness, ind.genome)
