from .abstract_individual import AbstractIndividual


class BinaryIndividual(AbstractIndividual):

    def __init__(self, fitness, genome, primitives):
        super().__init__(fitness, genome)

        self.primitives = primitives

    def clone(self):
        return BinaryIndividual(self.fitness, self.genome, self.primitives)

    def crossover(self, other_parent, rng):
        # perform crossover at random position in genome
        split_pos = rng.randint(len(self.genome))
        return BinaryIndividual(None, self.genome[:split_pos] + other_parent.genome[split_pos:], self.primitives)

    def mutate(self, mutation_rate, rng):

        n_mutations = int(mutation_rate * len(self.genome))
        assert n_mutations > 0

        for i in range(n_mutations):
            self.genome[rng.randint(len(self.genome))] = self.primitives[rng.randint(len(self.primitives))]  # mutate random gene

    def randomize_genome(self, genome_params, rng):
        self.genome = list(rng.choice(
            genome_params['primitives'],
            size=genome_params['genome_length']))

