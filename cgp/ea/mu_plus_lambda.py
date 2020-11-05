import concurrent.futures
from typing import Callable, List, Union

import numpy as np

from ..individual import IndividualBase
from ..population import Population


class MuPlusLambda:
    """Generic (mu + lambda) evolution strategy based on Deb et al. (2002).

    Currently only uses a single objective.

    Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002).
    A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE
    transactions on evolutionary computation, 6(2), 182-197.
    """

    def __init__(
        self,
        n_offsprings: int,
        tournament_size: int,
        *,
        n_processes: int = 1,
        local_search: Callable[[IndividualBase], None] = lambda combined: None,
        k_local_search: Union[int, None] = None,
        reorder_genome: bool = False,
    ):
        """Init function

        Parameters
        ----------
        n_offsprings : int
            Number of offspring in each iteration.
        tournament_size : int
            Tournament size in each iteration.
        n_processes : int, optional
            Number of parallel processes to be used. If greater than 1,
            parallel evaluation of the objective is supported. Defaults to 1.
        local_search : Callable[[Individua], None], optional
            Called before each fitness evaluation with a joint list of
            offsprings and parents to optimize numeric leaf values of
            the graph. Defaults to identity function.
        k_local_search : int
            Number of individuals in the whole population (parents +
            offsprings) to apply local search to.
       reorder_genome: bool, optional
            Whether genome reordering should be applied.
            Reorder shuffles the genotype of an individual without changing its phenotype,
            thereby contributing to neutral drift through the genotypic search space.
            If True, reorder is applied to each parents genome at every generation
            before creating offsprings.
            Defaults to True.
        """
        self.n_offsprings = n_offsprings

        self.tournament_size = tournament_size
        self.n_processes = n_processes
        self.local_search = local_search
        self.k_local_search = k_local_search
        self.reorder_genome = reorder_genome

        self.n_objective_calls: int = 0

    def initialize_fitness_parents(
        self, pop: Population, objective: Callable[[IndividualBase], IndividualBase]
    ) -> None:
        """Initialize the fitness of all parents in the given population.

        Parameters
        ----------
        pop : Population
            Population instance.
        objective : Callable[[gp.IndividualBase], gp.IndividualBase]
            An objective function used for the evolution. Needs to take an
            individual (IndividualBase) as input parameter and return
            a modified individual (with updated fitness).
        """
        # TODO can we avoid this function? how should a population be
        # initialized?
        pop._parents = self._compute_fitness(pop.parents, objective)

    def step(
        self, pop: Population, objective: Callable[[IndividualBase], IndividualBase],
    ) -> Population:
        """Perform one step in the evolution.

        Parameters
        ----------
        pop : Population
            Population instance.
        objective : Callable[[gp.IndividualBase], gp.IndividualBase]
            An objective function used for the evolution. Needs to take an
            individual (IndividualBase) as input parameter and return
            a modified individual (with updated fitness).

        Returns
        ----------
        Population
            Modified population with new parents.
        """

        if self.reorder_genome:
            pop.reorder_genome()

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

        # we follow a two-step process for selection of new parents:
        # we first determine the fitness for all individuals, then, if
        # applicable, we apply local search to the k_local_search
        # fittest individuals; after this we need to recompute the
        # fitness for all individuals for which parameters changed
        # during local search; finally we sort again by fitness, now
        # taking into account the effect of local search for
        # subsequent selection
        combined = self._compute_fitness(combined, objective)
        combined = self._sort(combined)

        n_total = self.n_offsprings + pop.n_parents
        k_local_search = n_total if self.k_local_search is None else self.k_local_search
        for idx in range(k_local_search):
            self.local_search(combined[idx])

        combined = self._compute_fitness(combined, objective)
        combined = self._sort(combined)

        pop.parents = self._create_new_parent_population(pop.n_parents, combined)

        return pop

    def _create_new_offspring_generation(self, pop: Population) -> List[IndividualBase]:
        # use tournament selection to randomly select individuals from
        # parent population
        offsprings: List[IndividualBase] = []
        while len(offsprings) < self.n_offsprings:
            tournament_pool = pop.rng.permutation(pop.parents)[: self.tournament_size]
            best_in_tournament = sorted(tournament_pool, key=lambda x: -x.fitness)[0]
            offsprings.append(best_in_tournament.clone())

        # mutate individuals to create offsprings
        offsprings = pop.mutate(offsprings)

        for ind in offsprings:
            ind.idx = pop.get_idx_for_new_individual()

        return offsprings

    def _compute_fitness(
        self, combined: List[IndividualBase], objective: Callable[[IndividualBase], IndividualBase]
    ) -> List[IndividualBase]:

        self.update_n_objective_calls(combined)

        # computes fitness on all individuals, objective functions
        # should return immediately if fitness is not None
        if self.n_processes == 1:
            combined = list(map(objective, combined))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                combined = list(executor.map(objective, combined))

        return combined

    def _sort(self, combined: List[IndividualBase]) -> List[IndividualBase]:
        def sort_func(ind: IndividualBase) -> float:
            """Return fitness of an individual, return -infinity for an individual
            with fitness equal nan, or raise error if the fitness is
            not a float.

            """
            if np.isnan(ind.fitness):
                return -np.inf

            if isinstance(ind.fitness, float):
                return ind.fitness
            else:
                raise ValueError(
                    f"IndividualBase fitness value is of wrong type {type(ind.fitness)}."
                )

        return sorted(combined, key=sort_func, reverse=True)

    def _create_new_parent_population(
        self, n_parents: int, combined: List[IndividualBase]
    ) -> List[IndividualBase]:
        """Create the new parent population by picking the first `n_parents`
        individuals from the combined population.

        """
        return combined[:n_parents]

    def update_n_objective_calls(self, combined: List[IndividualBase]) -> None:
        """Increase n_objective_calls by the number of individuals with fitness=None,
         i.e., for which the objective function will be evaluated.
        """
        for individual in combined:
            if individual.fitness is None:
                self.n_objective_calls += 1
