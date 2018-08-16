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
        self._primitives = primitives

    def randomize(self, levels_back):

        self._dna = []
        for i in range(self._n_regions):
            region = []
            region.append(self._primitives.sample())

            if i % self._n_rows == 0:  # only compute permissable inputs once per column
                current_column = i // self._n_rows
                permissable_inputs = self._permissable_inputs(current_column, levels_back)

            region += list(np.random.choice(permissable_inputs, self._length_per_region - 1))

            self._dna += region

        self._dna += self._random_input_for_output()

    def _permissable_inputs(self, column, levels_back):
        permissable_inputs = []
        permissable_inputs += [j for j in range(-self._n_inputs, 0)]
        permissable_inputs += [j for j in range(self._n_rows * max(0, (column - levels_back)), self._n_rows * (column))]
        return permissable_inputs

    def _random_input_for_output(self):
        return list(np.random.choice(range(-self._n_inputs, self._n_rows * self._n_columns), self._n_outputs))

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

    def _is_output_gene(self, idx):
        return idx >= (self._n_regions * self._length_per_region)

    def _is_function_gene(self, idx):
        return (idx % self._length_per_region) == 0

    def mutate(self, n_mutations, levels_back):

        for i in np.random.randint(0, len(self), n_mutations):

            # TODO: parameters to control mutation rates of specific
            # genes?
            if self._is_output_gene(i):
                self._dna[i] = self._random_input_for_output()[0]
            elif self._is_function_gene(i):
                self._dna[i] = self._primitives.sample()
            else:
                current_column = i // self._length_per_region // self._n_rows
                permissable_inputs = self._permissable_inputs(current_column, levels_back)
                self._dna[i] = np.random.choice(permissable_inputs)
