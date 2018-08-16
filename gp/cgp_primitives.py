import numpy as np


class CGPPrimitives():
    _primitives = None

    def __init__(self, primitives):
        self._primitives = {}
        for i in range(len(primitives)):
            self._primitives[i] = primitives[i]

    @property
    def max_arity(self):

        # TODO: determine arity without creating object instance
        arity = self._primitives[0](None, None).arity

        for idx, p in self._primitives.items():
            if arity < p(None, None).arity:
                arity = p(None, None).arity

        return arity

    def sample(self):
        return np.random.choice(list(self._primitives.keys()))

    def __getitem__(self, key):
        return self._primitives[key]
