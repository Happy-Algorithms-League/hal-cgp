import concurrent.futures
import numpy as np


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

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed, *, n_threads=1):

        self._n_parents = n_parents  # number of individuals in parent population
        self._n_offsprings = n_offsprings  # number of individuals in offspring population
        self._n_breeding = n_breeding  # size of breeding population
        self._tournament_size = tournament_size  # size of tournament for selection breeding population

        if not (0. < mutation_rate and mutation_rate < 1.):
            raise ValueError('mutation rate needs to be in (0, 1)')
        self._mutation_rate = mutation_rate  # probability of mutation per gene

        self._parents = None  # list of parent individuals
        self._offsprings = None  # list of offspring individuals
        self._combined = None  # list of all individuals

        self._max_idx = -1  # keeps track of maximal idx in population used to label individuals

        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.n_threads = n_threads  # number of threads to use for evaluating fitness

    def __getitem__(self, idx):
        return self._parents[idx]

    def generate_random_parent_population(self):
        self._parents = self._generate_random_individuals(self._n_parents, self.rng)
        self._label_new_individuals(self._parents)

    def generate_random_offspring_population(self):
        self._offsprings = self._generate_random_individuals(self._n_offsprings, self.rng)
        self._label_new_individuals(self._offsprings)

    def _label_new_individuals(self, individuals):
        for ind in individuals:
            ind.idx = self._max_idx + 1
            self._max_idx += 1

    def create_combined_population(self):

        # we want to prefer offsprings with the same fitness as their
        # parents as this allows accumulation of silent mutations;
        # since the built-in sort is stable (does not change the order
        # of elements that compare equal)
        # (https://docs.python.org/3.5/library/functions.html#sorted),
        # we can make sure that offsprings preceed parents with
        # identical fitness in the /sorted/ combined population by
        # concatenating the parent population to the offspring
        # population instead of the other way around
        self._combined = self._offsprings + self._parents

    def compute_fitness(self, objective, *, label=None):

        if label is not None:
            tmp_objective = lambda x: objective(x, label=label)
        else:
            tmp_objective = objective

        # computes fitness on all individuals, objective functions
        # should return immediately if fitness is not None
        if self.n_threads == 1:
            self._combined = list(map(tmp_objective, self._combined))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_threads) as executor:
                self._combined = list(executor.map(tmp_objective, self._combined))

    def sort(self):

        # replace all nan by -inf to make sure they end up at the end
        # after sorting
        for ind in self._combined:
            if np.isnan(ind.fitness):
                ind.fitness = -np.inf

        self._combined = sorted(self._combined, key=lambda x: -x.fitness)

    def create_new_parent_population(self):
        self._parents = []
        for i in range(self._n_parents):
            new_individual = self._combined[i].clone()

            # since this individual is genetically identical to its
            # parent, the identifier is the same
            new_individual.idx = self._combined[i].idx

            self._parents.append(new_individual)

    def create_new_offspring_population(self):

        # fill breeding pool via tournament selection from parent
        # population
        breeding_pool = []
        while len(breeding_pool) < self._n_breeding:
            tournament_pool = self.rng.permutation(self._parents)[:self._tournament_size]
            best_in_tournament = sorted(tournament_pool, key=lambda x: -x.fitness)[0]
            breeding_pool.append(best_in_tournament.clone())

        offsprings = self._crossover(breeding_pool, self.rng)
        offsprings = self._mutate(offsprings, self.rng)

        self._offsprings = offsprings

        self._label_new_individuals(self._offsprings)

    @property
    def champion(self):
        return sorted(self._parents, key=lambda x: -x.fitness)[0]

    def evolve(self, objective, max_generations, min_fitness, *, label=None):

        # generate initial parent population of size N
        self.generate_random_parent_population()

        # generate initial offspring population of size N
        self.generate_random_offspring_population()

        # perform evolution
        history_fitness = np.empty((max_generations, self._n_parents))
        history_dna_parents = np.empty((max_generations, self._n_parents, len(self._parents[0].genome.dna)))
        for generation in range(max_generations):

            # combine parent and offspring populations
            self.create_combined_population()

            #  compute fitness for all objectives for all individuals
            self.compute_fitness(objective, label=label)

            # sort population according to fitness & crowding distance
            self.sort()

            # fill new parent population according to sorting
            self.create_new_parent_population()

            # generate new offspring population from parent population
            self.create_new_offspring_population()

            # perform local search to tune values of constants
            # TODO pop.local_search(objective)

            history_fitness[generation] = self.fitness_parents()
            history_dna_parents[generation] = self.dna_parents()

            if np.mean(self.fitness_parents()) + 1e-10 >= min_fitness:
                break

        return history_fitness[:generation + 1], history_dna_parents[:generation + 1]

    def _generate_random_individuals(self, n, rng):
        raise NotImplementedError()

    def _crossover(self, breeding_pool):
        raise NotImplementedError()

    def _mutate(self, offsprings):
        raise NotImplementedError()

    def local_search(self, objective):
        raise NotImplementedError()

    @property
    def parents(self):
        return self._parents

    def fitness_parents(self):
        return [ind.fitness for ind in self._parents]

    def dna_parents(self):
        dnas = np.empty((self._n_parents, len(self._parents[0].genome)))
        for i in range(self._n_parents):
            dnas[i] = self._parents[i].genome.dna
        return dnas
