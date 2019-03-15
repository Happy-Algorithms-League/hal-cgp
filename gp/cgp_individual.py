from .abstract_individual import AbstractIndividual
from .cgp_genome import CGPGenome
from .cgp_graph import CGPGraph


class CGPIndividual(AbstractIndividual):

    def __init__(self, fitness, genome):
        super().__init__(fitness, genome)

    def clone(self):
        return CGPIndividual(self.fitness, self.genome.clone())

    def mutate(self, mutation_rate, rng):

        n_mutations = int(mutation_rate * len(self.genome.dna))
        assert n_mutations > 0

        graph = CGPGraph(self.genome)
        active_regions = graph.determine_active_regions()
        only_silent_mutations = self.genome.mutate(n_mutations, active_regions, rng)

        if not only_silent_mutations:
            self.fitness = None

    def randomize_genome(self, genome_params, rng):
        self.genome = CGPGenome(**genome_params)
        self.genome.randomize(rng)

