import numpy as np

from typing import List

from .individual import Individual, IndividualMultiGenome


class Population:
    """
    A population of individuals.
    """

    def __init__(
        self, n_parents: int, mutation_rate: float, seed: int, genome_params: dict
    ) -> None:
        """Init function.

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
        self.n_parents = n_parents  # number of individuals in parent population

        if not (0.0 < mutation_rate and mutation_rate < 1.0):
            raise ValueError("mutation rate needs to be in (0, 1)")
        self._mutation_rate = mutation_rate  # probability of mutation per gene

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self._genome_params = genome_params

        self._parents: List[Individual] = []  # list of parent individuals

        # keeps track of the number of generations, increases with
        # every new offspring generation
        self.generation = 0
        self._max_idx = -1  # keeps track of maximal idx in population used to label individuals

        self._generate_random_parent_population()

    @property
    def champion(self) -> Individual:
        """Return parent with the highest fitness.
        """
        return max(self._parents, key=lambda ind: ind.fitness)

    @property
    def parents(self) -> List[Individual]:
        return self._parents

    @parents.setter
    def parents(self, new_parents: List[Individual]) -> None:
        self.generation += 1
        self._parents = new_parents

    def __getitem__(self, idx: int) -> Individual:
        return self._parents[idx]

    def _generate_random_parent_population(self) -> None:
        self._parents = self._generate_random_individuals(self.n_parents)
        self._label_new_individuals(self._parents)

    def _label_new_individuals(self, individuals: List[Individual]) -> None:
        for ind in individuals:
            ind.idx = self._max_idx + 1
            self._max_idx += 1

    def _generate_random_individuals(self, n: int) -> List[Individual]:
        individuals = []
        for i in range(n):
            if isinstance(self._genome_params, dict):
                individual = Individual(fitness=None, genome=None)
            elif isinstance(self._genome_params, list) and isinstance(
                self._genome_params[0], dict
            ):
                individual = IndividualMultiGenome(fitness=None, genome=None)
            else:
                raise NotImplementedError()
            individual.randomize_genome(self._genome_params, self.rng)
            individuals.append(individual)

        return individuals

    def crossover(self, breeding_pool: List[Individual], n_offsprings: int) -> List[Individual]:
        """Create an offspring population via crossover.

        Parameters
        ----------
        breeding_pool : List[Individual]
            List of individuals from which the offspring are created.
        n_offsprings : int
            Number of offspring to be created.

        Returns
        ----------
        List[Individual]
            List of offspring individuals.
        """
        # in principle crossover would rely on a procedure like the
        # following:
        # offsprings = []
        # while len(offsprings) < n_offsprings:
        #     first_parent, second_parent = self.rng.permutation(breeding_pool)[:2]
        #     offsprings.append(first_parent.crossover(second_parent, self.rng))

        # return offsprings
        # however, as cross over tends to disrupt the search in in CGP
        # (Miller, 1999) crossover is skipped, instead the best
        # individuals from breeding pool are returned.
        # reference:
        # Miller, J. F. (1999). An empirical study of the efficiency
        # of learning boolean functions using a cartesian genetic
        # programming approach. In Proceedings of the 1st Annual
        # Conference on Genetic and Evolutionary Computation-Volume 2,
        # pages 1135–1142. Morgan Kaufmann Publishers Inc.
        assert len(breeding_pool) >= n_offsprings
        return sorted(breeding_pool, key=lambda x: -x.fitness)[:n_offsprings]

    def mutate(self, offsprings: List[Individual]) -> List[Individual]:
        """Mutate a list of offspring invididuals.

        Parameters
        ----------
        offsprings : List[Individual]
            List of offspring individuals to be mutated.

        Returns
        ----------
        List[Individual]
            List of mutated offspring individuals.
        """

        for off in offsprings:
            off.mutate(self._mutation_rate, self.rng)
        return offsprings

    def fitness_parents(self) -> List[float]:
        """Return fitness for all parents of the population.

        Returns
        ----------
        List[float]
            List of fitness values for all parents.
        """
        return [ind.fitness for ind in self._parents]

    def dna_parents(self) -> List[List[int]]:
        """Return a list of the DNA of all parents.

        Returns
        ----------
        List[List[int]]
            List of dna of all parents.
        """

        dnas = np.empty((self.n_parents, self._parents[0].genome._n_genes))
        for i in range(self.n_parents):
            dnas[i] = self._parents[i].genome.dna
        return dnas
