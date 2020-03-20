from .abstract_population import AbstractPopulation
from .cgp_individual import CGPIndividual, CGPIndividualMultiGenome


class CGPPopulation(AbstractPopulation):
    def __init__(self, n_parents, mutation_rate, seed, genome_params):
        """Init function.

        Extends AbstractPopulation.__init__


        Parameters
        ----------
        n_parents : int
            Number of parent individuals.
        mutation_rate : float
            Rate of mutations determining the number of genes to be
            mutated for offspring creation, between 0 and 1.
        seed : int
            Seed for internal random number generator.
        genome_params : dict
            Parameters for the genomes of the population's individuals.
        """
        self._genome_params = genome_params

        super().__init__(n_parents, mutation_rate, seed)

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            if isinstance(self._genome_params, dict):
                individual = CGPIndividual(fitness=None, genome=None)
            elif (isinstance(self._genome_params, list)
                  and isinstance(self._genome_params[0], dict)):
                individual = CGPIndividualMultiGenome(fitness=None, genome=None)
            else:
                raise NotImplementedError()
            individual.randomize_genome(self._genome_params, self.rng)
            individuals.append(individual)

        return individuals

    def crossover(self, breeding_pool, n_offsprings):
        assert len(breeding_pool) >= n_offsprings
        # do not perform crossover for CGP, just choose the best
        # individuals from breeding pool
        return sorted(breeding_pool, key=lambda x: -x.fitness)[:n_offsprings]
