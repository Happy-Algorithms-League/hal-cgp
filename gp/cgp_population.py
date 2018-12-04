import numpy as np
import torch

from .abstract_population import AbstractPopulation, Individual
from .cgp_genome import CGPGenome
from .cgp_graph import CGPGraph


class CGPIndividual(Individual):

    def __init__(self, fitness, genome):
        super().__init__(fitness, genome)

    def clone(self):
        return CGPIndividual(self.fitness, self.genome.clone())


class CGPPopulation(AbstractPopulation):

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate,
                 n_inputs, n_outputs, n_columns, n_rows, levels_back, primitives, *, n_threads=1):
        super().__init__(n_parents, n_offsprings, n_breeding, tournament_size, mutation_rate, n_threads=n_threads)

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
            genome = CGPGenome(self._n_inputs, self._n_outputs, self._n_columns, self._n_rows, self._levels_back, self._primitives)
            genome.randomize()
            individuals.append(CGPIndividual(fitness=None, genome=genome))
        return individuals

    def _crossover(self, breeding_pool):
        # do not perform crossover for CGP, just choose the best
        # individuals from breeding pool
        if len(breeding_pool) < self._n_offsprings:
            raise ValueError('size of breeding pool must be at least as large as the desired number of offsprings')
        return sorted(breeding_pool, key=lambda x: -x.fitness)[:self._n_offsprings]

    def _mutate(self, offsprings):

        n_mutations = int(self._mutation_rate * len(offsprings[0].genome))
        assert n_mutations > 0

        for off in offsprings:
            graph = CGPGraph(off.genome)
            active_regions = graph.determine_active_regions()
            only_silent_mutations = off.genome.mutate(n_mutations, active_regions)

            if not only_silent_mutations:
                off.fitness = None

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

    def compute_average_phenotype_distance_of_individuals(self):

        n_function_evaluations = 1000

        f_graph = [CGPGraph(ind.genome).compile_torch_class() for ind in self._parents]

        x = torch.Tensor(n_function_evaluations, self._n_inputs).normal_()

        d = 0
        for i, f_graph_i in enumerate(f_graph):
            for j, f_graph_j in enumerate(f_graph):
                if i != j:
                    d += torch.mean((f_graph_i(x) - f_graph_j(x)) ** 2)

        return 1. / self._n_parents * d

    def compute_average_genotype_distance_of_individuals(self):

        d = 0
        for i, ind_i in enumerate(self._parents):
            for j, ind_j in enumerate(self._parents):
                if i != j:
                    d += np.sum([not ind_i.genome[k] == ind_j.genome[k] for k in range(len(ind_i.genome))])

        return 1. / self._n_parents * d
