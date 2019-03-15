from .abstract_individual import AbstractIndividual


class BinaryIndividual(AbstractIndividual):

    def __init__(self, fitness, genome):
        super().__init__(fitness, genome)

    def clone(self):
        return BinaryIndividual(self.fitness, self.genome)

    def crossover(self, other_parent, rng):
        # perform crossover at random position in genome
        split_pos = rng.randint(len(self.genome))
        return BinaryIndividual(None, self.genome[:split_pos] + other_parent.genome[split_pos:])

    def mutate(self, mutation_rate, rng):

        n_mutations = int(mutation_rate * len(self.genome))
        assert n_mutations > 0

        for i in range(n_mutations):
            l_genome = list(self.genome)
            l_genome[rng.randint(len(self.genome))] = str(rng.randint(10))  # mutate random gene
            self.genome = ''.join(l_genome)

    def randomize_genome(self, genome_params, rng):
        self.genome = str(rng.randint(10 ** genome_params['genome_length'])).zfill(genome_params['genome_length'])

