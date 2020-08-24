from typing import List, Union

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
        mutation_rate: float,
        seed: int,
        genome_params: Union[dict, List[dict]],
    ) -> None:
        """Init function.

        Parameters
        ----------
        n_parents : int
            Number of parent individuals.
        mutation_rate : float
            Probability of a gene to be mutated, between 0 (excluded) and 1 (included).
        seed : int
            Seed for internal random number generator.
        genome_params : dict
            Parameters for the genomes of the population's individuals.
        """
        self.n_parents = n_parents  # number of individuals in parent population

        if not (0.0 < mutation_rate and mutation_rate <= 1.0):
            raise ValueError("mutation rate needs to be in (0, 1]")
        self._mutation_rate = mutation_rate  # probability of mutation per gene

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self._genome_params = genome_params

        self._parents: List[IndividualBase] = []  # list of parent individuals

        # keeps track of the number of generations, increases with
        # every new offspring generation
        self.generation = 0
        self._max_idx = 0  # keeps track of maximal idx in population used to label individuals

        self._generate_random_parent_population()

    @property
    def champion(self) -> IndividualBase:
        """Return parent with the highest fitness.
        """

        def key(ind: IndividualBase) -> float:
            assert isinstance(ind.fitness, float)
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

    def _generate_random_parent_population(self) -> None:
        parents: List[IndividualBase] = []
        for _ in range(self.n_parents):
            parents.append(self.generate_random_individual())
        self._parents = parents

    def get_idx_for_new_individual(self) -> int:
        idx = self._max_idx
        self._max_idx += 1
        return idx

    def generate_random_individual(self) -> IndividualBase:
        if isinstance(self._genome_params, dict):
            genome: Genome = Genome(**self._genome_params)
            individual_s = IndividualSingleGenome(
                fitness=None, genome=genome
            )  # type: IndividualBase # indicates to mypy that
            # individual_s is instance of a child class of
            # IndividualBase
            ind = individual_s
        else:
            genomes: List[Genome] = [Genome(**gd) for gd in self._genome_params]
            individual_m = IndividualMultiGenome(
                fitness=None, genome=genomes
            )  # type: IndividualBase # indicates to mypy that
            # individual_m is an instance of a child class of
            # IndividualBase
            ind = individual_m
        ind.randomize_genome(self.rng)
        ind.idx = self.get_idx_for_new_individual()
        ind.parent_idx = -1
        return ind

    def mutate(self, offsprings: List[IndividualBase]) -> List[IndividualBase]:
        """Mutate a list of offspring invididuals.

        Parameters
        ----------
        offsprings : List[IndividualBase]
            List of offspring individuals to be mutated.

        Returns
        ----------
        List[IndividualBase]
            List of mutated offspring individuals.
        """
        for off in offsprings:
            off.mutate(self._mutation_rate, self.rng)
        return offsprings

    def fitness_parents(self) -> List[Union[None, float]]:
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
