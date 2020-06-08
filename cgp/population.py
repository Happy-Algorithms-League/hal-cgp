import numpy as np

from typing import List, Union

from .genome import Genome
from .individual import IndividualBase, IndividualSingleGenome, IndividualMultiGenome


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
            Rate of mutations determining the number of genes to be
            mutated for offspring creation, between 0 and 1.
        seed : int
            Seed for internal random number generator.
        genome_params : dict
            Parameters for the genomes of the population's individuals.
        """
        self.n_parents = n_parents  # number of individuals in parent population

        if not (0.0 < mutation_rate and mutation_rate < 1.0):
            raise ValueError("mutation rate needs to be in (0, 1)")
        self._mutation_rate = mutation_rate  # probability of mutation per gene

        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self._genome_params = genome_params

        self._parents: List[IndividualBase] = []  # list of parent individuals

        # keeps track of the number of generations, increases with
        # every new offspring generation
        self.generation = 0
        self._max_idx = -1  # keeps track of maximal idx in population used to label individuals

        self._generate_random_parent_population()

    @property
    def champion(self) -> IndividualBase:
        """Return parent with the highest fitness.
        """
        return max(self._parents, key=lambda ind: ind.fitness)

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
        self._parents = self._generate_random_individuals(self.n_parents)
        self._label_new_individuals(self._parents)

    def _label_new_individuals(self, individuals: List[IndividualBase]) -> None:
        for ind in individuals:
            ind.idx = self._max_idx + 1
            self._max_idx += 1

    def _generate_random_individuals(self, n: int) -> List[IndividualBase]:
        individuals = []
        for i in range(n):
            if isinstance(self._genome_params, dict):
                genome: Genome = Genome(**self._genome_params)
                genome.randomize(self.rng)
                individual_s = IndividualSingleGenome(
                    fitness=None, genome=genome
                )  # type: IndividualBase # indicates to mypy that
                # individual_s is instance of a child class of
                # IndividualBase
                individuals.append(individual_s)
            else:
                genomes: List[Genome] = [Genome(**gd) for gd in self._genome_params]
                for g in genomes:
                    g.randomize(self.rng)
                individual_m = IndividualMultiGenome(
                    fitness=None, genome=genomes
                )  # type: IndividualBase # indicates to mypy that
                # individual_m is an instance of a child class of
                # IndividualBase
                individuals.append(individual_m)
        return individuals

    def crossover(
        self, breeding_pool: List[IndividualBase], n_offsprings: int
    ) -> List[IndividualBase]:
        """Create an offspring population via crossover.

        Parameters
        ----------
        breeding_pool : List[IndividualBase]
            List of individuals from which the offspring are created.
        n_offsprings : int
            Number of offspring to be created.

        Returns
        ----------
        List[IndividualBase]
            List of offspring individuals.
        """
        # in principle crossover would rely on a procedure like the
        # following:
        # offsprings = []
        # while len(offsprings) < n_offsprings:
        #     first_parent, second_parent = self.rng.permutation(breeding_pool)[:2]
        #     offsprings.append(first_parent.crossover(second_parent, self.rng))

        # return offsprings
        # however, as cross over tends to disrupt the search in in CGP
        # (Miller, 1999) crossover is skipped, instead the best
        # individuals from breeding pool are returned.
        # reference:
        # Miller, J. F. (1999). An empirical study of the efficiency
        # of learning boolean functions using a cartesian genetic
        # programming approach. In Proceedings of the 1st Annual
        # Conference on Genetic and Evolutionary Computation-Volume 2,
        # pages 1135–1142. Morgan Kaufmann Publishers Inc.
        assert len(breeding_pool) >= n_offsprings

        def sort_func(ind: IndividualBase) -> float:
            if isinstance(ind.fitness, float):
                return ind.fitness
            else:
                raise ValueError(f"Individual fitness value is of wrong type {type(ind.fitness)}.")

        # Sort individuals in descending order
        return sorted(breeding_pool, key=sort_func, reverse=True)[:n_offsprings]

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
