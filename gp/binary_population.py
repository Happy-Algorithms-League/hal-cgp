from .abstract_population import AbstractPopulation
from .binary_individual import BinaryIndividual, BinaryGenome


class BinaryPopulation(AbstractPopulation):
    def __init__(self, n_parents, mutation_rate, seed, genome_params):
        self._genome_params = genome_params
        super().__init__(n_parents, mutation_rate, seed)

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            genome = BinaryGenome(**self._genome_params)
            individual = BinaryIndividual(fitness=None, genome=genome)
            individual.randomize_genome(self.rng)
            individuals.append(individual)

        return individuals
