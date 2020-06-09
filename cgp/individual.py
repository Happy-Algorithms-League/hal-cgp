import numpy as np
from typing import Callable, List, Union

try:
    import sympy  # noqa: F401
    from sympy.core import expr as sympy_expr  # noqa: F401

    sympy_available = True
except ModuleNotFoundError:
    sympy_available = False

try:
    import torch  # noqa: F401

    torch_available = True
except ModuleNotFoundError:
    torch_available = False

from .genome import Genome
from .cartesian_graph import CartesianGraph


class IndividualBase:
    """Base class for all individuals.
    """

    def __init__(self, fitness: Union[float, None]) -> None:
        """Init function.

        fitness : float
            Fitness of the individual.
        """
        self.fitness: Union[float, None] = fitness
        self.idx: int

    def clone(self):
        raise NotImplementedError()

    def mutate(self, mutation_rate, rng):
        raise NotImplementedError()

    def randomize_genome(self, rng):
        raise NotImplementedError()

    def to_func(self):
        raise NotImplementedError()

    def to_numpy(self):
        raise NotImplementedError()

    def to_torch(self):
        raise NotImplementedError()

    def to_sympy(self, simplify):
        raise NotImplementedError()

    def update_parameters_from_torch_class(self, torch_cls):
        raise NotImplementedError()

    @staticmethod
    def _mutate_genome(genome: Genome, mutation_rate: float, rng: np.random.RandomState) -> bool:
        """Mutate a given genome.

        Parameters
        ----------
        genome : Genome
            Genome to be mutated.
        mutation_rate : float
            Proportion of genes to be mutated, between 0 and 1.
        rng : numpy.RandomState
            Random number generator instance.

        Returns
        -------
        bool
            Whether all mutations were silent.
        """
        n_mutations = int(mutation_rate * len(genome.dna))
        assert n_mutations > 0

        graph = CartesianGraph(genome)
        active_regions = graph.determine_active_regions()
        only_silent_mutations = genome.mutate(n_mutations, active_regions, rng)

        return only_silent_mutations

    @staticmethod
    def _randomize_genome(genome: Genome, rng: np.random.RandomState) -> None:
        """Randomize the individual's genome.

        Parameters
        ----------
        genome : Genome
            Genome to be randomized.
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        -------
        None
        """
        genome.randomize(rng)

    @staticmethod
    def _to_func(genome: Genome) -> Callable[[List[float]], List[float]]:
        return CartesianGraph(genome).to_func()

    @staticmethod
    def _to_numpy(genome: Genome) -> Callable[[np.ndarray], np.ndarray]:
        return CartesianGraph(genome).to_numpy()

    @staticmethod
    def _to_torch(genome: Genome) -> "torch.nn.Module":
        return CartesianGraph(genome).to_torch()

    @staticmethod
    def _to_sympy(genome: Genome, simplify) -> "sympy_expr.Expr":
        return CartesianGraph(genome).to_sympy(simplify)

    @staticmethod
    def _update_parameters_from_torch_class(genome: Genome, torch_cls: "torch.nn.Module") -> bool:
        return genome.update_parameters_from_torch_class(torch_cls)


class IndividualSingleGenome(IndividualBase):
    """An individual representing a particular computational graph.
    """

    def __init__(self, fitness: Union[float, None], genome: Genome) -> None:
        """Init function.

        fitness : float
            Fitness of the individual.
        genome: Genome
            Genome of the individual.
        """
        super().__init__(fitness)
        self.genome: Genome = genome

    def __repr__(self):
        return f"Individual(idx={self.idx}, fitness={self.fitness}, genome={self.genome}))"

    def clone(self) -> "IndividualSingleGenome":
        return IndividualSingleGenome(self.fitness, self.genome.clone())

    def mutate(self, mutation_rate: float, rng: np.random.RandomState) -> None:
        only_silent_mutations = self._mutate_genome(self.genome, mutation_rate, rng)
        if not only_silent_mutations:
            self.fitness = None

    def randomize_genome(self, rng: np.random.RandomState) -> None:
        self._randomize_genome(self.genome, rng)

    def to_func(self) -> Callable[[List[float]], List[float]]:
        return self._to_func(self.genome)

    def to_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        return self._to_numpy(self.genome)

    def to_torch(self) -> "torch.nn.Module":
        return self._to_torch(self.genome)

    def to_sympy(self, simplify: bool = True) -> "sympy_expr.Expr":
        return self._to_sympy(self.genome, simplify)

    def update_parameters_from_torch_class(self, torch_cls: "torch.nn.Module") -> None:
        any_parameter_updated = self._update_parameters_from_torch_class(self.genome, torch_cls)
        if any_parameter_updated:
            self.fitness = None


class IndividualMultiGenome(IndividualBase):
    """An individual with multiple genomes each representing a particular computational graph.
    """

    def __init__(self, fitness: Union[float, None], genome: List[Genome]) -> None:
        """Init function.

        fitness : float
            Fitness of the individual.
        genome: List[Genome]
            List of genomes of the individual.
        """
        super().__init__(fitness)
        self.genome: List[Genome] = genome

    def clone(self) -> "IndividualMultiGenome":
        return IndividualMultiGenome(self.fitness, [g.clone() for g in self.genome])

    def mutate(self, mutation_rate: float, rng: np.random.RandomState) -> None:
        for g in self.genome:
            only_silent_mutations = self._mutate_genome(g, mutation_rate, rng)
            if not only_silent_mutations:
                self.fitness = None

    def randomize_genome(self, rng: np.random.RandomState) -> None:
        for g in self.genome:
            self._randomize_genome(g, rng)

    def to_func(self) -> List[Callable[[List[float]], List[float]]]:
        return [self._to_func(g) for g in self.genome]

    def to_numpy(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [self._to_numpy(g) for g in self.genome]

    def to_torch(self) -> List["torch.nn.Module"]:
        return [self._to_torch(g) for g in self.genome]

    def to_sympy(self, simplify: bool = True) -> List["sympy_expr.Expr"]:
        return [self._to_sympy(g, simplify) for g in self.genome]

    def update_parameters_from_torch_class(self, torch_cls: List["torch.nn.Module"]) -> None:
        any_parameter_updated = any(
            [
                self._update_parameters_from_torch_class(g, tcls)
                for g, tcls in zip(self.genome, torch_cls)
            ]
        )
        if any_parameter_updated:
            self.fitness = None
