from .cgp_genome import CGPGenome
from .cgp_graph import CGPGraph


class CGPIndividual:
    """An individual representing a particular computational graph.
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
        return f"Individual(idx={self.idx}, fitness={self.fitness}, genome={self.genome}))"

    def clone(self):
        """Clone the individual.

        Returns
        -------
        gp.CGPIndividual
        """
        return CGPIndividual(self.fitness, self.genome.clone())

    def crossover(self, other_parent, rng):
        """Create a new individual by cross over with another individual.

        Parameters
        ----------
        other_parent : gp.CGPIndividual
            Other individual to perform crossover with.
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        -------
        gp.CGPIndividual
        """
        raise NotImplementedError("crossover currently not supported")

    def _mutate(self, genome, mutation_rate, rng):

        n_mutations = int(mutation_rate * len(genome.dna))
        assert n_mutations > 0

        graph = CGPGraph(genome)
        active_regions = graph.determine_active_regions()
        only_silent_mutations = genome.mutate(n_mutations, active_regions, rng)

        if not only_silent_mutations:
            self.fitness = None

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
        self._mutate(self.genome, mutation_rate, rng)

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
        self.genome = CGPGenome(**genome_params)
        self.genome.randomize(rng)

    def to_func(self):
        """Return the expression represented by the individual as Callable.

        Returns
        ----------
        Callable
        """
        return CGPGraph(self.genome).to_func()

    def to_sympy(self, simplify=True):
        """Return the expression represented by the individual as SymPy
        expression.

        Returns
        ----------
        SymPy expression
        """
        return CGPGraph(self.genome).to_sympy(simplify)


class CGPIndividualMultiGenome(CGPIndividual):
    """An individual with multiple genomes.

    Derived from gp.CGPIndividual.
    """

    def clone(self):
        return CGPIndividualMultiGenome(self.fitness, [g.clone() for g in self.genome])

    def mutate(self, mutation_rate, rng):
        for g in self.genome:
            self._mutate(g, mutation_rate, rng)

    def randomize_genome(self, genome_params, rng):
        self.genome = []
        for g_params in genome_params:
            self.genome.append(CGPGenome(**g_params))
            self.genome[-1].randomize(rng)

    def to_func(self):
        """Return list of Callables from the individual's genomes.

        Returns
        ----------
        List[Callable]
        """

        return [CGPGraph(g).to_func() for g in self.genome]

    def to_sympy(self, simplify=True):
        """Return list of str expressions defining the functions represented by the invidual's genomes.

        Returns
        ----------
        str
        """

        return [CGPGraph(g).to_sympy(simplify) for g in self.genome]
