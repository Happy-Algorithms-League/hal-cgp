import numpy as np
import torch

from .abstract_population import AbstractPopulation, Individual
from .cgp_genome import CGPGenome
from .cgp_graph import CGPGraph


class CGPPopulation(AbstractPopulation):

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, n_mutations,
                 n_inputs, n_outputs, n_columns, n_rows, levels_back, primitives):
        super().__init__(n_parents, n_offsprings, n_breeding, tournament_size, n_mutations)

        if len(primitives) == 0:
            raise RuntimeError('need to provide at least one function primitive')

        self._n_inputs = n_inputs
        self._n_hidden = n_columns * n_rows
        self._n_outputs = n_outputs

        self._n_columns = n_columns
        self._n_rows = n_rows
        self._levels_back = levels_back

        self._primitives = primitives

    def _generate_random_individuals(self, n):
        individuals = []
        for i in range(n):
            genome = CGPGenome(self._n_inputs, self._n_outputs, self._n_columns, self._n_rows, self._primitives)
            genome.randomize(self._levels_back)
            individuals.append(Individual(None, genome))
        return individuals

    def _crossover(self, breeding_pool):
        # do not perform crossover for CGP, just choose the best
        # individuals from breeding pool
        return sorted(breeding_pool, key=lambda x: -x.fitness)[:self._n_offsprings]

    def _mutate(self, offsprings):
        for off in offsprings:
            off.genome.mutate(self._n_mutations, self._levels_back)

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

    def _clone_individual(self, ind):
        return Individual(ind.fitness, ind.genome.clone())
