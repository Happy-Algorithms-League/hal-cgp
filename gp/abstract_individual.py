class AbstractIndividual():
    """
    Generic individual class for evolutionary algorithms. Provides container
    for fitness and genome. Derived classes need to define how individuals
    should be cloned.

    """

    def __init__(self, fitness, genome):
        self.fitness = fitness
        self.genome = genome

        self.idx = None

    def __repr__(self):
        return 'Individual(idx={}, fitness={}, genome={}))'.format(self.idx, self.fitness, self.genome)

    def clone(self):
        raise NotImplementedError()
