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

    def mutate(self, n_mutations, rng):
        for i in range(n_mutations):
            l_genome = list(self.genome)
            l_genome[rng.randint(len(self.genome))] = str(rng.randint(10))  # mutate random gene
            self.genome = ''.join(l_genome)

    def randomize_genome(self, rng):
        self.genome = str(rng.randint(10 ** len(self.genome))).zfill(len(self.genome))

