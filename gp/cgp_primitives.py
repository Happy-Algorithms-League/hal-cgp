import numpy as np

from .cgp_node import CGPAdd, CGPSub


class CGPPrimitives():
    _primitives = None

    def __init__(self):
        self._primitives = {}
        self._primitives[0] = CGPAdd
        self._primitives[1] = CGPSub

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
