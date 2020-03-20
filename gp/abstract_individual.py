class AbstractIndividual():
    """
    Generic individual class for evolutionary algorithms. Provides container
    for fitness and genome. Derived classes need to define how individuals
    should be cloned, crossovered, mutated and randomized.

    """

    def __init__(self, fitness, genome):
        """Init function.

        fitness : float
            Fitness of the individual.
        genome: Genome instance
            Genome of the invididual.
        """
        self.fitness = fitness
        self.genome = genome
        self.idx = None  # an identifier to keep track of all unique genomes

    def __repr__(self):
        return f'Individual(idx={self.idx}, fitness={self.fitness}, genome={self.genome}))'

    def clone(self):
        """Clone the individual.

        Returns
        -------
        gp.AbstractIndividual
        """
        raise NotImplementedError()

    def crossover(self, other_parent, rng):
        """Create a new individual by cross over with another individual.

        Parameters
        ----------
        other_parent : gp.AbstractIndividual
            Other individual to perform crossover with.
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        -------
        gp.AbstractIndividual
        """
        raise NotImplementedError()

    def mutate(self, mutation_rate, rng):
        """Mutate the individual in place.
        
        Parameters
        ----------
        mutation_rate : float
            Proportion of mutations determining the number of genes to be mutated, between 0 and 1.
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def randomize_genome(self, genome_params, rng):
        """Randomize the individual's genome.

        Parameters
        ----------
        genome_params : dict
            Parameter dictionary for the new randomized genome.
            Needs to contain: n_inputs, n_outputs, n_columns, n_rows,
            levels_back, primitives.
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        ----------
        None
        """
        raise NotImplementedError()
