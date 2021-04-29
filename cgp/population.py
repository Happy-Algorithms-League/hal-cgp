from typing import Callable, List, Optional, Union

import numpy as np

from .genome import Genome
from .individual import IndividualBase, IndividualMultiGenome, IndividualSingleGenome


class Population:
    """
    A population of individuals.
    """

    def __init__(
        self,
        n_parents: int,
        seed: int,
        genome_params: Union[dict, List[dict]],
        individual_init: Optional[Callable[[IndividualBase], IndividualBase]] = None,
    ) -> None:
        """Init function.

        Parameters
        ----------
        n_parents : int
            Number of parent individuals.
        seed : int
            Seed for internal random number generator.
        genome_params : dict
            Parameters for the genomes of the population's individuals.
        individual_init: callable, optional
            If not None, called for each individual of the initial
            parent population, for example, to set the dna of parents.
        """
        self.n_parents = n_parents  # number of individuals in parent population

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self._genome_params = genome_params

        self._parents: List[IndividualBase] = []  # list of parent individuals

        # keeps track of the number of generations, increases with
        # every new offspring generation
        self.generation = 0
        self._max_idx = 0  # keeps track of maximal idx in population used to label individuals

        if individual_init is not None and not callable(individual_init):
            raise TypeError("individual_init must be a callable")

        self._generate_random_parent_population(individual_init)

    @property
    def champion(self) -> IndividualBase:
        """Return parent with the highest fitness.
        """

        def key(ind: IndividualBase) -> float:
            return ind.fitness

        return max(self._parents, key=key)

    @property
    def parents(self) -> List[IndividualBase]:
        return self._parents

    @parents.setter
    def parents(self, new_parents: List[IndividualBase]) -> None:
        self.generation += 1
        self._parents = new_parents

    def __getitem__(self, idx: int) -> IndividualBase:
        return self._parents[idx]

    def _generate_random_parent_population(
        self, individual_init: Optional[Callable[[IndividualBase], IndividualBase]] = None
    ) -> None:
        parents: List[IndividualBase] = []
        for _ in range(self.n_parents):
            ind = self.generate_random_individual()
            if individual_init is not None:
                ind = individual_init(ind)
            parents.append(ind)
        self._parents = parents

    def get_idx_for_new_individual(self) -> int:
        idx = self._max_idx
        self._max_idx += 1
        return idx

    def generate_random_individual(self) -> IndividualBase:
        if isinstance(self._genome_params, dict):
            genome: Genome = Genome(**self._genome_params)
            individual_s = IndividualSingleGenome(
                genome=genome
            )  # type: IndividualBase # indicates to mypy that
            # individual_s is instance of a child class of
            # IndividualBase
            ind = individual_s
        else:
            genomes: List[Genome] = [Genome(**gd) for gd in self._genome_params]
            individual_m = IndividualMultiGenome(
                genome=genomes
            )  # type: IndividualBase # indicates to mypy that
            # individual_m is an instance of a child class of
            # IndividualBase
            ind = individual_m
        ind.randomize_genome(self.rng)
        ind.idx = self.get_idx_for_new_individual()
        ind.parent_idx = -1
        return ind

    def fitness_parents(self) -> List[Optional[float]]:
        """Return fitness for all parents of the population.

        Returns
        ----------
        List[float]
            List of fitness values for all parents.
        """
        return [ind.fitness for ind in self._parents]

    def reorder_genome(self) -> None:
        """ Reorders the genome for all parents in the population

        Returns
        ---------
        None
        """
        for parent in self.parents:
            parent.reorder_genome(self.rng)
