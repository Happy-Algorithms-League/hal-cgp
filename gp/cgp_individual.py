from .abstract_individual import AbstractIndividual

class CGPIndividual(AbstractIndividual):

    def __init__(self, fitness, genome):
        super().__init__(fitness, genome)

    def clone(self):
        return CGPIndividual(self.fitness, self.genome.clone())
