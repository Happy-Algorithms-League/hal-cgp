""" Generic (mu + lambda) evolution strategy based on Deb et al. (2002).

    Population class needs to implement functions to
    - generate random individuals
    - perform crossover
    - perform mutations
    - [perform local search]

    Currently only uses a single objective.

    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II.
    IEEE transactions on evolutionary computation, 6(2), 182-197.
"""

import concurrent.futures
import numpy as np


class MuPlusLambda():

    def __init__(self, n_parents, n_offsprings, n_breeding, tournament_size, *, n_processes=1):
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings

        if n_breeding < n_offsprings:
            raise ValueError('size of breeding pool must be at least as large as the desired number of offsprings')
        self.n_breeding = n_breeding

        self.tournament_size = tournament_size
        self.n_processes = n_processes

    def initialize_fitness_parents(self, pop, objective, *, label=None):
        # TODO can we avoid this function? how should a population be
        # initialized?
        pop._parents = self._compute_fitness(pop, objective, label=label)

    def step(self, pop, objective, *, label=None):
        # create new offspring generation
        offsprings = self._create_new_offspring_generation(pop)

        # we want to prefer offsprings with the same fitness as their
        # parents as this allows accumulation of silent mutations;
        # since the built-in sort is stable (does not change the order
        # of elements that compare equal)
        # (https://docs.python.org/3.5/library/functions.html#sorted),
        # we can make sure that offsprings preceed parents with
        # identical fitness in the /sorted/ combined population by
        # concatenating the parent population to the offspring
        # population instead of the other way around
        combined = offsprings + pop.parents

        # compute fitness
        combined = self._compute_fitness(combined, objective, label=label)

        # sort
        combined = self._sort(combined)

        # create new parent population
        pop.parents = self._create_new_parent_population(combined)

        return pop

    def _create_new_offspring_generation(self, pop):
        # fill breeding pool via tournament selection from parent
        # population
        breeding_pool = []
        while len(breeding_pool) < self.n_breeding:
            tournament_pool = pop.rng.permutation(pop.parents)[:self.tournament_size]
            best_in_tournament = sorted(tournament_pool, key=lambda x: -x.fitness)[0]
            breeding_pool.append(best_in_tournament.clone())

        # create offsprings by applying crossover to breeding pool and mutating
        # resulting individuals
        offsprings = pop.crossover(breeding_pool, self.n_offsprings)
        offsprings = pop.mutate(offsprings)
        # TODO this call to pop to label individual is quite ugly, find a
        # better way to track ids of individuals; maybe in the EA?
        pop._label_new_individuals(offsprings)

        return offsprings

    def _compute_fitness(self, combined, objective, *, label=None):
        if label is not None:
            tmp_objective = lambda x: objective(x, label=label)
        else:
            tmp_objective = objective

        # computes fitness on all individuals, objective functions
        # should return immediately if fitness is not None
        if self.n_processes == 1:
            combined = list(map(tmp_objective, combined))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                combined = list(executor.map(tmp_objective, combined))

        return combined

    def _sort(self, combined):
        # replace all nan by -inf to make sure they end up at the end
        # after sorting
        for ind in combined:
            if np.isnan(ind.fitness):
                ind.fitness = -np.inf

        combined = sorted(combined, key=lambda x: -x.fitness)

        return combined

    def _create_new_parent_population(self, combined):

        # create new parent population by picking the `n_parents` individuals
        # with the highest fitness
        parents = []
        for i in range(self.n_parents):
            new_individual = combined[i].clone()

            # since this individual is genetically identical to its
            # parent, the identifier is the same
            new_individual.idx = combined[i].idx

            parents.append(new_individual)

        return parents
