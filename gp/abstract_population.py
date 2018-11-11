import numpy as np


class Individual():
    fitness = None
    genome = None

    def __init__(self, fitness, genome):
        self.fitness = fitness
        self.genome = genome

    def __repr__(self):
        return 'Individual(fitness={}, genome={})'.format(self.fitness, self.genome)


class AbstractPopulation():
    """
    Generic population class for evolutionary algorithms based on Deb et al. (2002).

    Derived class need to implement functions to
    - generate random individuals
    - perform crossover
    - perform mutations
    - perform local search

    Currently only uses a single objective.

    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.
    """

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, n_mutations):

        self._n_parents = n_parents  # number of individuals in parent population
        self._n_offsprings = n_offsprings  # number of individuals in offspring population
        self._n_breeding = n_breeding  # size of breeding population
        self._tournament_size = tournament_size  # size of tournament for selection breeding population
        self._n_mutations = n_mutations  # number of mutations in genome per individual

        self._parents = None  # list of parent individuals
        self._offsprings = None  # list of offspring individuals
        self._combined = None  # list of all individuals

    def __getitem__(self, idx):
        return self._parents[idx]

    def generate_random_parent_population(self):
        self._parents = self._generate_random_individuals(self._n_parents)

    def generate_random_offspring_population(self):
        self._offsprings = self._generate_random_individuals(self._n_offsprings)

    def create_combined_population(self):
        self._combined = self._parents + self._offsprings

    def compute_fitness(self, objective):

        for ind in self._combined:
            fitness = objective(ind.genome)
            ind.fitness = fitness

    def sort(self):
        self._combined = sorted(self._combined, key=lambda x: -x.fitness)

    def create_new_parent_population(self):
        self._parents = []
        for i in range(self._n_parents):
            self._parents.append(self._clone_individual(self._combined[i]))

    def create_new_offspring_population(self):

        # fill breeding pool via tournament selection from parent
        # population
        breeding_pool = []
        while len(breeding_pool) < self._n_breeding:
            tournament_pool = np.random.permutation(self._parents)[:self._tournament_size]
            best_in_tournament = sorted(tournament_pool, key=lambda x: -x.fitness)[0]
            breeding_pool.append(self._clone_individual(best_in_tournament))

        offsprings = self._crossover(breeding_pool)
        offsprings = self._mutate(offsprings)

        self._offsprings = offsprings

    @property
    def champion(self):
        return sorted(self._parents, key=lambda x: -x.fitness)[0]

    def _generate_random_individuals(self, n):
        raise NotImplementedError()

    def _crossover(self, breeding_pool):
        raise NotImplementedError()

    def _mutate(self, offsprings):
        raise NotImplementedError()

    def local_search(self, objective):
        raise NotImplementedError()

    def _clone_individual(self, ind):
        raise NotImplementedError()

    @property
    def parents(self):
        return self._parents

    @property
    def fitness(self):
        return [ind.fitness for ind in self._parents]
