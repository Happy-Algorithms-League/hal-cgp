import numpy as np


class Individual():
    fitness = None
    genome = None

    def __init__(self, fitness, genome):
        self.fitness = fitness
        self.genome = genome

    def __repr__(self):
        return 'Individual(fitness={}, genome={})'.format(self.fitness, self.genome)


class Population():
    _n_individuals = None  # number of individuals in parent population
    _genome_length = None  # length of genome
    _n_breeding = None  # size of breeding population
    _tournament_size = None  # size of tournament for selection breeding population
    _n_mutations = None  # number of mutations in genome per individual

    _parents = None  # list of parent individuals
    _offsprings = None  # list of offspring individuals
    _combined = None  # list of all individuals

    def __init__(self, n_individuals, genome_length, n_breeding, tournament_size, n_mutations):
        self._n_individuals = n_individuals
        self._genome_length = genome_length
        self._n_breeding = n_breeding
        self._tournament_size = tournament_size
        self._n_mutations = n_mutations

    def _generate_random_individuals(self):
        individuals = []
        for i in range(self._n_individuals):
            individuals.append(
                Individual(None, str(np.random.randint(10 ** self._genome_length)).zfill(self._genome_length)))
        return individuals

    def generate_random_parent_population(self):
        self._parents = self._generate_random_individuals()

    def generate_random_offspring_population(self):
        self._offsprings = self._generate_random_individuals()

    def compute_fitness(self, objective):
        self._combined = []
        for ind in self._parents:
            self._combined.append(Individual(objective(ind.genome), ind.genome))
        for ind in self._offsprings:
            self._combined.append(Individual(objective(ind.genome), ind.genome))

    def sort(self):
        self._combined = sorted(self._combined, key=lambda x: -x.fitness)

    def create_new_parent_population(self):
        self._parents = []
        for i in range(self._n_individuals):
            self._parents.append(self._combined[i])

    def create_new_offspring_population(self):
        # fill breeding pool via tournament selection
        breeding_pool = []
        while len(breeding_pool) < self._n_breeding:
            sample = sorted(np.random.permutation(self._combined)[:self._tournament_size], key=lambda x: -x.fitness)
            breeding_pool.append(sample[0])

        offsprings = self._crossover(breeding_pool)
        offsprings = self._mutate(offsprings)

        self._offsprings = offsprings

    def _crossover(self, breeding_pool):
        offsprings = []
        while len(offsprings) < self._n_individuals:
            # choose parents and perform crossover at random position in genome
            parents = np.random.permutation(breeding_pool)[:2]
            split_pos = np.random.randint(self._genome_length)
            offsprings.append(
                Individual(None, parents[0].genome[:split_pos] + parents[1].genome[split_pos:]))

        return offsprings

    def _mutate(self, offsprings):
        for off in offsprings:
            for i in range(self._n_mutations):
                # mutate random gene
                genome = list(off.genome)
                genome[np.random.randint(self._genome_length)] = str(np.random.randint(10))
                off.genome = ''.join(genome)

        return offsprings

    @property
    def parents(self):
        return self._parents

    @property
    def fitness(self):
        return [ind.fitness for ind in self._parents]
