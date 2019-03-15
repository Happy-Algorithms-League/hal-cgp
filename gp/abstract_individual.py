class AbstractIndividual():
    """
    Generic individual class for evolutionary algorithms. Provides container
    for fitness and genome. Derived classes need to define how individuals
    should be cloned, crossovered, mutated and randomized.

    """

    def __init__(self, fitness, genome):
        self.fitness = fitness
        self.genome = genome
        self.idx = None  # an identifier to keep track of all unique genomes

    def __repr__(self):
        return 'Individual(idx={}, fitness={}, genome={}))'.format(self.idx, self.fitness, self.genome)

    def clone(self):
        raise NotImplementedError()

    def crossover(self, other_parent, rng):
        raise NotImplementedError()

    def mutate(self, mutation_rate, rng):
        raise NotImplementedError()

    def randomize_genome(self, genome_params, rng):
        raise NotImplementedError()
