import numpy as np
import torch

from .abstract_population import AbstractPopulation
from .cgp_genome import CGPGenome
from .cgp_graph import CGPGraph
from .cgp_individual import CGPIndividual, CGPIndividualMultiGenome


class CGPPopulation(AbstractPopulation):

    def __init__(self, n_parents, mutation_rate, seed, genome_params):

        self._genome_params = genome_params

        super().__init__(n_parents, mutation_rate, seed)

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            if isinstance(self._genome_params, dict):
                individual = CGPIndividual(fitness=None, genome=None)
            elif isinstance(self._genome_params, list) and isinstance(self._genome_params[0], dict):
                individual = CGPIndividualMultiGenome(fitness=None, genome=None)
            else:
                raise NotImplementedError()
            individual.randomize_genome(self._genome_params, self.rng)
            individuals.append(individual)

        return individuals

    def crossover(self, breeding_pool, n_offsprings):
        assert len(breeding_pool) >= n_offsprings
        # do not perform crossover for CGP, just choose the best
        # individuals from breeding pool
        return sorted(breeding_pool, key=lambda x: -x.fitness)[:n_offsprings]

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
