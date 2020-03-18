import numpy as np


class AbstractPopulation():
    """
    Generic population class for evolutionary algorithms.
    """

    def __init__(self, n_parents, mutation_rate, seed):
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
        """
        self.n_parents = n_parents  # number of individuals in parent population

        if not (0. < mutation_rate and mutation_rate < 1.):
            raise ValueError('mutation rate needs to be in (0, 1)')
        self._mutation_rate = mutation_rate  # probability of mutation per gene

        self._parents = None  # list of parent individuals

        self.generation = 0  # keeps track of the number of generations, increases with every new offspring generation
        self._max_idx = -1  # keeps track of maximal idx in population used to label individuals

        self.seed = seed
        self.rng = np.random.RandomState(seed)

        self._generate_random_parent_population()

    @property
    def champion(self):
        """Return parent with the highest fitness.
        """
        return max(self._parents, key=lambda ind: ind.fitness)

    @property
    def parents(self):
        return self._parents

    @parents.setter
    def parents(self, new_parents):
        self.generation += 1
        self._parents = new_parents

    def __getitem__(self, idx):
        return self._parents[idx]

    def _generate_random_parent_population(self):
        self._parents = self._generate_random_individuals(self.n_parents)
        self._label_new_individuals(self._parents)

    def _label_new_individuals(self, individuals):
        for ind in individuals:
            ind.idx = self._max_idx + 1
            self._max_idx += 1

    def _generate_random_individuals(self, n):
        raise NotImplementedError()

    def crossover(self, breeding_pool, n_offsprings):
        """Create an offspring population via crossover.

        Parameters
        ----------
        breeding_pool : List[gp.AbstractPopulation]
            List of individuals from which the offspring are created.
        n_offsprings : int
            Number of offspring to be created.

        Returns
        ----------
        List[gp.AbstractIndividual]
            List of offspring individuals.
        """
        offsprings = []
        while len(offsprings) < n_offsprings:
            first_parent, second_parent = self.rng.permutation(breeding_pool)[:2]
            offsprings.append(first_parent.crossover(second_parent, self.rng))

        return offsprings

    def mutate(self, offsprings):
        """Mutate a list of offspring invididuals.

        Parameters
        ----------
        offsprings : List[gp.AbstractIndividual]
            List of offspring individuals to be mutated.

        Returns
        ----------
        List[gp.AbstractIndividual]
            List of mutated offspring individuals.
        """

        for off in offsprings:
            off.mutate(self._mutation_rate, self.rng)
        return offsprings

    def fitness_parents(self):
        """Return fitness for all parents of the population.

        Returns
        ----------
        List[float]
            List of fitness values for all parents.
        """
        return [ind.fitness for ind in self._parents]

    def dna_parents(self):
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
