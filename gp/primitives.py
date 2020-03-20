import numpy as np

from typing import Iterator, List, Tuple, Type

from .node import Node


class Primitives:
    """Class collecting primitives of the Cartesian Genetic Programming framework.
    """

    _max_arity = 0
    _primitives: dict = {}

    def __init__(self, primitives: List[Type[Node]]) -> None:
        """Init function.

        Parameters
        ----------
        primitives : List[Type[Node]]
            List of primitives.
        """
        for i in range(len(primitives)):
            if not isinstance(primitives[i], type):
                raise TypeError(f"expected class but received {type(primitives[i])}")
            if not issubclass(primitives[i], Node):
                raise TypeError(f"expected subclass of Node but received {primitives[i].__name__}")

        self._primitives = {}
        for i in range(len(primitives)):
            self._primitives[i] = primitives[i]

        # hide primitives dict behind MappingProxyType to make sure it
        # is not changed after construction
        # unfortunately not supported by pickle, necessary for
        # multiprocessing; another way to implement this?
        # self._primitives = types.MappingProxyType(self._primitives)

        self._determine_max_arity()

    def __iter__(self) -> Iterator:
        return iter([self[i] for i in range(len(self._primitives))])

    def _determine_max_arity(self) -> None:

        arity = 1  # minimal possible arity (output nodes need one input)

        for idx, p in self._primitives.items():
            if arity < p._arity:
                arity = p._arity

        self._max_arity = arity

    def sample(self, rng: np.random.RandomState) -> int:
        """Sample a random primitive.

        Parameters
        ----------
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        -------
        int
            Index of the sample primitive
        """
        return rng.choice(self.alleles)

    def __getitem__(self, key: int) -> Type[Node]:
        return self._primitives[key]

    @property
    def max_arity(self) -> int:
        return self._max_arity

    @property
    def alleles(self) -> Tuple:
        return tuple(self._primitives.keys())

    def __len__(self):
        return len(self._primitives)
