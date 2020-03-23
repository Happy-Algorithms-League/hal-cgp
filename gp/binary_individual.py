from .abstract_individual import AbstractIndividual


class BinaryGenome:
    def __init__(self, genome_length, primitives, p_primitives=None):

        self.dna = None
        self.genome_length = genome_length
        self.primitives = primitives
        self.p_primitives = p_primitives

    def __len__(self):
        return self.genome_length

    def __repr__(self):
        return self.__class__.__name__ + "(" + "".join(str(d) for d in self.dna) + ")"

    def __getitem__(self, key):
        return self.dna[key]

    def clone(self):
        new = BinaryGenome(self.genome_length, self.primitives, self.p_primitives)
        new.dna = list(self.dna)
        return new

    def randomize(self, rng):
        self.dna = list(rng.choice(self.primitives, size=self.genome_length))

    def mutate(self, n_mutations, rng):
        for i in range(n_mutations):
            if self.p_primitives:
                # mutate random gene according to distribution of primitives
                self.dna[rng.randint(self.genome_length)] = rng.choice(
                    self.primitives, p=self.p_primitives
                )
            else:
                self.dna[rng.randint(self.genome_length)] = rng.choice(
                    self.primitives
                )  # mutate random gene


class BinaryIndividual(AbstractIndividual):
    def __init__(self, fitness, genome):
        super().__init__(fitness, genome)

    def clone(self):
        return BinaryIndividual(self.fitness, self.genome.clone())

    def crossover(self, other_parent, rng):
        # perform crossover at random position in genome
        split_pos = rng.randint(len(self.genome))
        new_genome = self.genome.clone()
        new_genome.dna = self.genome.dna[:split_pos] + other_parent.genome.dna[split_pos:]
        return BinaryIndividual(None, new_genome)

    def mutate(self, mutation_rate, rng):

        n_mutations = int(mutation_rate * len(self.genome))
        assert n_mutations > 0

        self.genome.mutate(n_mutations, rng)

    def randomize_genome(self, rng):
        self.genome.randomize(rng)
