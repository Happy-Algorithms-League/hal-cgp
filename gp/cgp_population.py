import numpy as np
import torch

from .abstract_population import AbstractPopulation
from .cgp_genome import CGPGenome
from .cgp_graph import CGPGraph
from .cgp_individual import CGPIndividual



class CGPPopulation(AbstractPopulation):

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed,
                 genome_params, *, n_threads=1):
        super().__init__(n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, seed, n_threads=n_threads)

        self._genome_params = genome_params

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            individual = CGPIndividual(fitness=None, genome=None)
            individual.randomize_genome(self._genome_params, self.rng)
            individuals.append(individual)

        return individuals

    def _crossover(self, breeding_pool):
        # do not perform crossover for CGP, just choose the best
        # individuals from breeding pool
        if len(breeding_pool) < self._n_offsprings:
            raise ValueError('size of breeding pool must be at least as large as the desired number of offsprings')
        return sorted(breeding_pool, key=lambda x: -x.fitness)[:self._n_offsprings]

    def _mutate(self, offsprings):

        n_mutations = int(self._mutation_rate * offsprings[0].genome._n_genes)
        assert n_mutations > 0

        for off in offsprings:
            off.mutate(n_mutations, self.rng)

        return offsprings

    # def local_search(self, objective):

    #     for off in self._offsprings:

    #         graph = CGPGraph(off.genome)
    #         f = graph.compile_torch_class()

    #         if len(list(f.parameters())) > 0:
    #             optimizer = torch.optim.SGD(f.parameters(), lr=1e-1)
    #             criterion = torch.nn.MSELoss()

    #         history_loss_trial = []
    #         history_loss_bp = []
    #         for j in range(100):
    #             x = torch.Tensor(2).normal_()
    #             y = f(x)
    #             loss = objective()
    #             history_loss_trial.append(loss.detach().numpy())

    #             if len(list(f.parameters())) > 0:
    #                 y_target = 2.7182 + x[0] - x[1]

    #                 loss = criterion(y[0], y_target)
    #                 f.zero_grad()
    #                 loss.backward()
    #                 optimizer.step()

    #                 history_loss_bp.append(loss.detach().numpy())

    #         graph.update_parameter_values(f)
