import numpy as np


class CGPGenome():
    _n_regions = None
    _length_per_region = None
    _dna = None

    def __init__(self, n_inputs, n_outputs, n_columns, n_rows, primitives, levels_back):
        self._n_regions = n_columns * n_rows
        self._length_per_region = 1 + primitives.max_arity

        self._dna = []
        for i in range(self._n_regions):
            region = []
            region.append(primitives.sample())

            # TODO: compute this only once for each column
            current_column = i // n_rows + 1
            permissable_inputs = []
            permissable_inputs += [j for j in range(-n_inputs, 0)]
            permissable_inputs += [j for j in range(n_rows * max(0, (current_column - 1 - levels_back)), n_rows * (current_column - 1))]

            region += list(np.random.choice(permissable_inputs, self._length_per_region - 1))

            self._dna += region

        # self._dna += list(np.random.choice(range(-n_inputs, n_rows * n_columns), n_outputs))
        # TODO: allow direct connections from input
        self._dna += list(np.random.choice(range(0, n_rows * n_columns), n_outputs))

    def __iter__(self):
        for i in range(self._n_regions):
            yield self._dna[i * self._length_per_region:(i + 1) * self._length_per_region]

    def __getitem__(self, idx):
        return self._dna[idx]
