import numpy as np


class CGPGenome():
    _n_regions = None
    _length_per_region = None
    _dna = None

    def __init__(self, n_inputs, n_outputs, n_columns, n_rows, primitives):
        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._n_columns = n_columns
        self._n_rows = n_rows
        self._n_regions = n_columns * n_rows
        self._length_per_region = 1 + primitives.max_arity

    def randomize(self, primitives, levels_back):

        self._dna = []
        for i in range(self._n_regions):
            region = []
            region.append(primitives.sample())

            if i % self._n_rows == 0:  # only compute permissable inputs once per column
                current_column = i // self._n_rows
                permissable_inputs = []
                permissable_inputs += [j for j in range(-self._n_inputs, 0)]
                permissable_inputs += [j for j in range(self._n_rows * max(0, (current_column - levels_back)), self._n_rows * (current_column))]

            region += list(np.random.choice(permissable_inputs, self._length_per_region - 1))

            self._dna += region

        self._dna += list(np.random.choice(range(-self._n_inputs, self._n_rows * self._n_columns), self._n_outputs))

    def __iter__(self):
        if self._dna is None:
            raise RuntimeError('dna not initialized - call CGPGenome.randomize first')
        for i in range(self._n_regions):
            yield self._dna[i * self._length_per_region:(i + 1) * self._length_per_region]

    def __getitem__(self, key):
        if self._dna is None:
            raise RuntimeError('dna not initialized - call CGPGenome.randomize first')
        return self._dna[key]

    def __len__(self):
        return self._n_regions * self._length_per_region + self._n_outputs

    @property
    def dna(self):
        return self._dna

    @dna.setter
    def dna(self, value):
        assert(len(value) == len(self))
        self._dna = value
