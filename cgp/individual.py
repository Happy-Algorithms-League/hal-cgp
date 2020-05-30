from abc import ABCMeta, abstractmethod
import numpy as np
from typing import Any, Callable, List, Union

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


def mutate_genome(genome: Genome, mutation_rate: float, rng: np.random.RandomState) -> bool:
    """Mutate a given genome.

    Parameters
    ----------
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


class IndividualBase(metaclass=ABCMeta):
    """Base class for all individuals.
    """

    def __init__(self, fitness: Union[float, None], genome: Union[Genome, List[Genome]]) -> None:
        """Init function.

        fitness : float
            Fitness of the individual.
        """
        self.fitness = fitness
        self.genome: Union[Genome, List[Genome]]
        self.idx: int

    def __repr__(self):
        return f"Individual(idx={self.idx}, fitness={self.fitness}, genome={self.genome}))"

    def clone(self) -> "IndividualBase":
        """Clone the individual.

        Returns
        -------
        Individual
        """
        return type(self)(self.fitness, self.genome)

    def crossover(self, other_parent: "IndividualBase", rng: np.random.RandomState) -> None:
        """Create a new individual by cross over with another individual.

        Parameters
        ----------
        other_parent : Individual
            Other individual to perform crossover with.
        rng : numpy.random.RandomState
            Random number generator instance to use for crossover.

        Returns
        -------
        Individual
        """
        raise NotImplementedError("crossover currently not supported")

    @abstractmethod
    def mutate(self, mutation_rate: float, rng: np.random.RandomState) -> None:
        """Mutate the individual in place.

        Parameters
        ----------
        mutation_rate : float
            Proportion of genes to be mutated, between 0 and 1.
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def randomize_genome(self, rng: np.random.RandomState) -> None:
        """Randomize the individual's genome.

        Parameters
        ----------
        genome_params : dict
            Parameter dictionary for the new randomized genome.
            Needs to contain: n_inputs, n_outputs, n_columns, n_rows,
            levels_back, primitives.
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        -------
        None
        """
        pass

    @abstractmethod
    def to_func(
        self
    ) -> Union[Callable[[List[float]], List[float]], List[Callable[[List[float]], List[float]]]]:
        """Return the expression represented by the individual as Callable.

        Returns
        ----------
        Callable[[List[float]], List[float]]
        """
        pass

    @abstractmethod
    def to_numpy(
        self
    ) -> Union[Callable[[np.ndarray], np.ndarray], List[Callable[[np.ndarray], np.ndarray]]]:
        """Return the expression represented by the individual as
        NumPy-compatible Callable.

        Returns
        -------
        Callable[[np.ndarray], np.ndarray]:
        """
        pass

    @abstractmethod
    def to_torch(self) -> Union["torch.nn.Module", List["torch.nn.Module"]]:
        """Return the expression represented by the individual as Torch class.

        Returns
        -------
        torch.nn.Module
        """
        pass

    @abstractmethod
    def to_sympy(self, simplify: bool = True) -> Union["sympy_expr.Expr", List["sympy_expr.Expr"]]:
        """Return the expression represented by the individual as SymPy
        expression.

        Returns
        -------
        List[sympy_expr.Expr]:
        """
        pass

    @abstractmethod
    def update_parameters_from_torch_class(self, torch_cls: Any) -> None:
        """Update parameters of the individual from a torch class.
        """
        pass


class IndividualSingleGenome(IndividualBase):
    """An individual representing a particular computational graph.
    """

    def __init__(self, fitness: Union[float, None], genome: Genome) -> None:
        """Init function.

        fitness : float
            Fitness of the individual.
        genome: Genome instance
            Genome of the individual.
        """
        super().__init__(fitness, genome)
        self.genome: Genome = genome

    def mutate(self, mutation_rate: float, rng: np.random.RandomState) -> None:
        only_silent_mutations = mutate_genome(self.genome, mutation_rate, rng)
        if not only_silent_mutations:
            self.fitness = None

    def randomize_genome(self, rng: np.random.RandomState) -> None:
        self.genome.randomize(rng)

    def to_func(self) -> Callable[[List[float]], List[float]]:
        return CartesianGraph(self.genome).to_func()

    def to_numpy(self) -> Callable[[np.ndarray], np.ndarray]:
        return CartesianGraph(self.genome).to_numpy()

    def to_torch(self) -> "torch.nn.Module":
        return CartesianGraph(self.genome).to_torch()

    def to_sympy(self, simplify: bool = True) -> "sympy_expr.Expr":
        return CartesianGraph(self.genome).to_sympy(simplify)

    def update_parameters_from_torch_class(self, torch_cls: "torch.nn.Module") -> None:
        any_parameter_updated = self.genome.update_parameters_from_torch_class(torch_cls)
        if any_parameter_updated:
            self.fitness = None


class IndividualMultiGenome(IndividualBase):
    """An individual with multiple genomes each representing a particular computational graph.
    """

    def __init__(self, fitness: Union[float, None], genome: List[Genome]) -> None:
        """Init function.

        fitness : float
            Fitness of the individual.
        genome: Genome instance
            Genome of the individual.
        """
        super().__init__(fitness, genome)

        self.genome: List[Genome] = genome

    def mutate(self, mutation_rate: float, rng: np.random.RandomState) -> None:
        for g in self.genome:
            only_silent_mutations = mutate_genome(g, mutation_rate, rng)
            if not only_silent_mutations:
                self.fitness = None

    def randomize_genome(self, rng: np.random.RandomState) -> None:
        for g in self.genome:
            g.randomize(rng)

    def to_func(self) -> List[Callable[[List[float]], List[float]]]:
        return [CartesianGraph(g).to_func() for g in self.genome]

    def to_numpy(self) -> List[Callable[[np.ndarray], np.ndarray]]:
        return [CartesianGraph(g).to_numpy() for g in self.genome]

    def to_torch(self) -> List["torch.nn.Module"]:
        return [CartesianGraph(g).to_torch() for g in self.genome]

    def to_sympy(self, simplify: bool = True) -> List["sympy_expr.Expr"]:
        return [CartesianGraph(g).to_sympy(simplify) for g in self.genome]

    def update_parameters_from_torch_class(self, torch_cls: List["torch.nn.Module"]) -> None:
        any_parameter_updated = any(
            [g.update_parameters_from_torch_class(tcls) for g, tcls in zip(self.genome, torch_cls)]
        )

        if any_parameter_updated:
            self.fitness = None
