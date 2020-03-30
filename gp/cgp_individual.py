from .abstract_individual import AbstractIndividual
from .cgp_genome import CGPGenome
from .cgp_graph import CGPGraph


class CGPIndividual(AbstractIndividual):
    """Individual representing a particular Cartesian computational graph.

    Derived from gp.AbstractIndividual.
    """

    def clone(self):
        new_individual = CGPIndividual(self.fitness, self.genome.clone())
        new_individual.parameter_names_to_values = self.parameter_names_to_values.copy()
        return new_individual

    def _mutate(self, genome, mutation_rate, rng):

        n_mutations = int(mutation_rate * len(genome.dna))
        assert n_mutations > 0

        graph = CGPGraph(genome)
        active_regions = graph.determine_active_regions()
        only_silent_mutations = genome.mutate(n_mutations, active_regions, rng)

        if not only_silent_mutations:
            self.fitness = None

    def mutate(self, mutation_rate, rng):
        self._mutate(self.genome, mutation_rate, rng)

    def randomize_genome(self, genome_params, rng):
        self.genome = CGPGenome(**genome_params)
        self.genome.randomize(rng)

    def to_func(self):
        """Return Callable from the individual's genome.

        Returns
        ----------
        Callable
        """
        return CGPGraph(self.genome).to_func(self.parameter_names_to_values)

    def to_sympy(self, simplify=True):
        """Return str expression defining the function represented by the invidual's genome.

        Returns
        ----------
        str
        """
        return CGPGraph(self.genome).to_sympy(simplify, self.parameter_names_to_values)

    def to_torch(self):
        return CGPGraph(self.genome).to_torch(self.parameter_names_to_values)

    def update_parameters_from_torch_class(self, torch_cls):
        """Update values stored in constant nodes of graph from parameters of a given Torch instance.

        Can be used to import new values from a Torch class after a autograd step.

        Parameters
        ----------
        torch_cls : torch.nn.module
            Instance of a torch class.

        Returns
        -------
        None
        """
        for name, value in torch_cls.named_parameters():
            name = f"<{name[1:]}>"
            if name in self.parameter_names_to_values:
                self.parameter_names_to_values[name] = value.item()


class CGPIndividualMultiGenome(CGPIndividual):
    """Individual of the Cartesian Genetic Programming framework with multiple genomes.

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
