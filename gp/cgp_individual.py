from .abstract_individual import AbstractIndividual
from .cgp_genome import CGPGenome
from .cgp_graph import CGPGraph


class CGPIndividual(AbstractIndividual):

    def __init__(self, fitness, genome):
        super().__init__(fitness, genome)

    def clone(self):
        return CGPIndividual(self.fitness, self.genome.clone())

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
        return CGPGraph(self.genome).to_func()

    def to_sympy(self, simplify=True):
        return CGPGraph(self.genome).to_sympy(simplify)


class CGPIndividualMultiGenome(CGPIndividual):

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

    def to_sympy(self, simplify=True):
        return [CGPGraph(g).to_sympy(simplify) for g in self.genome]
