import copy
from typing import Callable, List, Set, Type, Union

import numpy as np

from .cartesian_graph import CartesianGraph
from .genome import Genome

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


class IndividualBase:
    """Base class for all individuals.
    """

    __base_attrs__: Set[str]

    def __init__(self) -> None:
        """Init function.
        """
        self._fitness: List[Union[None, float]] = [None]
        self._objective_idx: int = 0

        self.idx: Union[int, None] = None
        self.parent_idx: Union[int, None] = None

    def __init_subclass__(cls: Type) -> None:

        # store the attributes present right after instance creation,
        # i.e., all attributes not set by the user
        cls.__base_attrs__ = set(cls(None).__dict__.keys())

    @property
    def fitness(self) -> float:
        return sum([f for f in self._fitness if f is not None])

    @fitness.setter
    def fitness(self, v: float) -> None:
        if not isinstance(v, float):
            raise ValueError(f"IndividualBase fitness value is of wrong type {type(v)}.")

        self._fitness[self._objective_idx] = v

    @property
    def fitness_current_objective(self) -> Union[None, float]:
        return self._fitness[self._objective_idx]

    @property
    def objective_idx(self) -> int:
        return self._objective_idx

    @objective_idx.setter
    def objective_idx(self, v: int) -> None:
        if len(self._fitness) <= v:
            self._fitness = list(self._fitness) + [None]
        self._objective_idx = v

    def clone(self):
        raise NotImplementedError()

    def _copy_user_defined_attributes(self, other):
        """Copy all attributes that are not defined in __init__ of the (sub
        and super) class from self to other.
        """
        for attr in self.__dict__:
            if attr not in self.__base_attrs__:
                setattr(other, attr, copy.deepcopy(getattr(self, attr)))

    def fitness_is_None(self) -> bool:
        return self._fitness[self._objective_idx] is None

    def mutate(self, mutation_rate, rng):
        raise NotImplementedError()

    def randomize_genome(self, rng):
        raise NotImplementedError()

    def reorder_genome(self, rng):
        raise NotImplementedError()

    def reset_fitness(self):
        for i in range(len(self._fitness)):
            self._fitness[i] = None

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
        return genome.mutate(mutation_rate, rng)

    @staticmethod
    def _randomize_genome(genome: Genome, rng: np.random.RandomState) -> None:
        genome.randomize(rng)

    @staticmethod
    def _reorder_genome(genome: Genome, rng: np.random.RandomState) -> None:
        genome.reorder(rng)

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

    def __lt__(self, other):
        for i in range(len(self._fitness)):
            if self._fitness[i] is None and other._fitness[i] is None:
                return False
            elif self._fitness[i] is not None and other._fitness[i] is None:
                return False
            elif self._fitness[i] is None and other._fitness[i] is not None:
                return True
            elif self._fitness[i] < other._fitness[i]:
                return True
        return False


class IndividualSingleGenome(IndividualBase):
    """An individual representing a particular computational graph.
    """

    def __init__(self, genome: Genome) -> None:
        """Init function.

        genome: Genome
            Genome of the individual.
        """
        super().__init__()
        self.genome: Genome = genome

    def __repr__(self):
        return f"Individual(idx={self.idx}, fitness={self.fitness}, genome={self.genome}))"

    def clone(self) -> "IndividualSingleGenome":
        ind = IndividualSingleGenome(self.genome.clone())
        ind._fitness = list(self._fitness)
        ind.parent_idx = self.idx
        self._copy_user_defined_attributes(ind)
        return ind

    def mutate(self, mutation_rate: float, rng: np.random.RandomState) -> None:
        only_silent_mutations = self._mutate_genome(self.genome, mutation_rate, rng)
        if not only_silent_mutations:
            self.reset_fitness()

    def randomize_genome(self, rng: np.random.RandomState) -> None:
        self._randomize_genome(self.genome, rng)

    def reorder_genome(self, rng: np.random.RandomState) -> None:
        self._reorder_genome(self.genome, rng)

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
            self.reset_fitness()


class IndividualMultiGenome(IndividualBase):
    """An individual with multiple genomes each representing a particular computational graph.
    """

    def __init__(self, genome: List[Genome]) -> None:
        """Init function.

        genome: List[Genome]
            List of genomes of the individual.
        """
        super().__init__()
        self.genome: List[Genome] = genome

    def clone(self) -> "IndividualMultiGenome":
        ind = IndividualMultiGenome([g.clone() for g in self.genome])
        ind._fitness = list(self._fitness)
        ind.parent_idx = self.idx
        self._copy_user_defined_attributes(ind)
        return ind

    def mutate(self, mutation_rate: float, rng: np.random.RandomState) -> None:
        for g in self.genome:
            only_silent_mutations = self._mutate_genome(g, mutation_rate, rng)
            if not only_silent_mutations:
                self.reset_fitness()

    def randomize_genome(self, rng: np.random.RandomState) -> None:
        for g in self.genome:
            self._randomize_genome(g, rng)

    def reorder_genome(self, rng: np.random.RandomState) -> None:
        for g in self.genome:
            self._reorder_genome(g, rng)

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
            self.reset_fitness()
