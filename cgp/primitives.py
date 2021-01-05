from dataclasses import dataclass, field
from typing import Iterator, Tuple, Type

import numpy as np

from .node import Node


@dataclass(frozen=True)
class Primitives:
    """Convenience class to manage primitives, i.e., Node classes.

    """

    _primitives: Tuple[Type[Node], ...]
    _max_arity: int = field(init=False)

    def __post_init__(self) -> None:
        self._check_types()
        self.__dict__[
            "_max_arity"
        ] = self._determine_max_arity()  # avoid using use __setattr_ since dataclass is frozen

    def _check_types(self) -> None:
        if not isinstance(self._primitives, tuple):
            raise TypeError(f"expected tuple but received {type(self._primitives)}")

        for i in range(len(self._primitives)):
            if not isinstance(self._primitives[i], type):
                raise TypeError(
                    f"expected class but received instance of {type(self._primitives[i])}"
                )
            if not issubclass(self._primitives[i], Node):
                raise TypeError(
                    f"expected subclass of Node but received class {self._primitives[i].__name__}"
                )

    def _determine_max_arity(self) -> int:

        arity = 1  # minimal possible arity (output nodes need one address)

        for p in self._primitives:
            if arity < p._arity:
                arity = p._arity

        return arity

    def __iter__(self) -> Iterator[Type[Node]]:
        return iter(self._primitives)

    def sample_allele(self, rng: np.random.RandomState) -> int:
        """Sample a random primitive index.

        Parameters
        ----------
        rng : numpy.RandomState
            Random number generator instance.

        Returns
        -------
        int
            Index of the sampled primitive.
        """
        return rng.randint(len(self._primitives))

    def __getitem__(self, key: int) -> Type[Node]:
        if key < 0 or key >= len(self._primitives):
            raise IndexError("primitive index out of bounds")
        return self._primitives[key]

    @property
    def max_arity(self) -> int:
        return self._max_arity

    def is_valid_allele(self, allele: int) -> bool:
        return (allele >= 0) and (allele < len(self._primitives))
