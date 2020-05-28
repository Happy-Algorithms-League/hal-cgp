import collections
import numpy as np

try:
    import torch  # noqa: F401

    torch_available = True
except ModuleNotFoundError:
    torch_available = False

from typing import DefaultDict, Generator, List, Optional, Tuple, Type

from .node import Node
from .primitives import Primitives


ID_INPUT_NODE: int = -1
ID_OUTPUT_NODE: int = -2
ID_NON_CODING_GENE: int = -3


def return_float_one() -> float:
    """Constructor for default value of defaultdict. Needs to be global to
    support pickling.

    """
    return 1.0


class Genome:
    """Genome class for  individuals.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_columns: int,
        n_rows: int,
        levels_back: int,
        primitives: Tuple[Type[Node], ...],
    ) -> None:
        """Init function.

        Parameters
        ----------
        n_inputs : int
            Number of inputs of the function represented by the genome.
        n_outputs : int
            Number of outputs of the function represented by the genome.
        n_columns : int
            Number of columns in the representation of the genome.
        n_rows : int
            Number of rows in the representation of the genome.
        levels_back : int
            Number of previous columns that an entry in the genome can be
            connected with.
        primitives : Tuple[Type[Node], ...]
           Tuple of primitives that the genome can refer to.
        """
        if n_inputs <= 0:
            raise ValueError("n_inputs must be strictly positive")
        self._n_inputs = n_inputs

        if n_columns < 0:
            raise ValueError("n_columns must be non-negative")
        self._n_columns = n_columns

        if n_rows < 0:
            raise ValueError("n_rows must be non-negative")
        self._n_rows = n_rows

        if n_outputs <= 0:
            raise ValueError("n_outputs must be strictly positive")
        self._n_outputs = n_outputs

        if levels_back is None:
            levels_back = n_columns
        if levels_back == 0:
            raise ValueError("levels_back must be strictly positive")
        if levels_back > n_columns:
            raise ValueError("levels_back can not be larger than n_columns")
        self._levels_back = levels_back

        self._primitives = Primitives(primitives)
        self._length_per_region = (
            1 + self._primitives.max_arity
        )  # one function gene + multiple input genes

        self._dna: List[int] = []  # stores dna as list of alleles for all regions

        # constants used as identifiers for input and output nodes
        self._id_input_node: int = ID_INPUT_NODE
        self._id_output_node: int = ID_OUTPUT_NODE
        self._id_unused_gene: int = ID_NON_CODING_GENE

        # dictionary to store values of Parameter nodes
        self.parameter_names_to_values: DefaultDict = collections.defaultdict(return_float_one)

    def __getitem__(self, key: int) -> int:
        if self._dna is None:
            raise RuntimeError("dna not initialized")
        return self._dna[key]

    def __setitem__(self, key: int, value: int) -> None:
        dna = list(self._dna)
        dna[key] = value
        self._validate_dna(dna)
        self._dna = dna

    @property
    def dna(self) -> List[int]:
        return self._dna

    @dna.setter
    def dna(self, value: List[int]) -> None:
        self._validate_dna(value)
        self._dna = value

    @property
    def _n_hidden(self) -> int:
        return self._n_columns * self._n_rows

    @property
    def _n_regions(self) -> int:
        return self._n_inputs + self._n_hidden + self._n_outputs

    @property
    def _n_genes(self) -> int:
        return self._n_regions * self._length_per_region

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        for region_idx, input_region in self.iter_input_regions():
            s += str(region_idx) + ": " + str(input_region) + " | "
        for region_idx, hidden_region in self.iter_hidden_regions():
            s += str(region_idx) + ": " + str(hidden_region) + " | "
        for region_idx, output_region in self.iter_output_regions():
            s += str(region_idx) + ": " + str(output_region) + " | "
        s = s[:-3]
        s += ")"
        return s

    def _create_input_region(self) -> List[int]:

        region = []
        region.append(self._id_input_node)
        # non-coding input genes since input nodes do not receive
        # inputs themselves
        region += [self._id_unused_gene] * self._primitives.max_arity
        return region

    def _create_random_hidden_region(
        self, rng: np.random.RandomState, permissible_inputs: List[int]
    ) -> List[int]:

        region = []
        node_id = self._primitives.sample_allele(rng)
        region.append(node_id)
        region += list(rng.choice(permissible_inputs, self._primitives.max_arity))

        return region

    def _create_random_output_region(
        self, rng: np.random.RandomState, permissible_inputs: List[int]
    ) -> List[int]:

        region = []
        region.append(self._id_output_node)
        region.append(rng.choice(permissible_inputs))
        # output nodes have only one input, other genes are hence non-coding
        region += [self._id_unused_gene] * (self._primitives.max_arity - 1)

        return region

    def randomize(self, rng: np.random.RandomState) -> None:
        """Randomize the genome.

        Parameters
        ----------
        rng : numpy.RandomState
            Random number generator instance to use for crossover.

        Returns
        ----------
        None
        """
        dna: List[int] = []

        # add input nodes
        for i in range(self._n_inputs):
            dna += self._create_input_region()

        # add hidden nodes
        for i in range(self._n_hidden):

            if i % self._n_rows == 0:  # only compute permissible inputs once per column
                permissible_inputs = self._permissible_inputs(i + self._n_inputs)

            dna += self._create_random_hidden_region(rng, permissible_inputs)

        # add output nodes
        permissible_inputs = self._permissible_inputs_for_output_region()  # identical for outputs
        for i in range(self._n_outputs):
            dna += self._create_random_output_region(rng, permissible_inputs)

        # accept generated dna if it is valid
        self._validate_dna(dna)
        self._dna = dna

    def _permissible_inputs(self, region_idx: int) -> List[int]:

        assert not self._is_input_region(region_idx)

        permissible_inputs = []

        # all nodes can be connected to input
        permissible_inputs += [j for j in range(0, self._n_inputs)]

        # add all nodes reachable according to levels back
        if self._is_hidden_region(region_idx):
            hidden_column_idx = self._hidden_column_idx(region_idx)
            lower = self._n_inputs + self._n_rows * max(0, (hidden_column_idx - self._levels_back))
            upper = self._n_inputs + self._n_rows * hidden_column_idx
        else:
            assert self._is_output_region(region_idx)
            lower = self._n_inputs
            upper = self._n_inputs + self._n_rows * self._n_columns

        permissible_inputs += [j for j in range(lower, upper)]

        return permissible_inputs

    def _permissible_inputs_for_output_region(self) -> List[int]:
        return self._permissible_inputs(self._n_inputs + self._n_rows * self._n_columns)

    def _validate_dna(self, dna: List[int]) -> None:

        if len(dna) != self._n_genes:
            raise ValueError("dna length mismatch")

        for region_idx, input_region in self.iter_input_regions(dna):

            if input_region[0] != self._id_input_node:
                raise ValueError(
                    "function genes for input nodes need to be identical to input node identifiers"
                )

            if input_region[1:] != ([self._id_unused_gene] * self._primitives.max_arity):
                raise ValueError("input genes for input nodes need to be empty")

        for region_idx, hidden_region in self.iter_hidden_regions(dna):

            if not self._primitives.is_valid_allele(hidden_region[0]):
                raise ValueError("function gene for hidden node has invalid value")

            input_genes = hidden_region[1:]

            permissible_inputs = set(self._permissible_inputs(region_idx))
            if not set(input_genes).issubset(permissible_inputs):
                raise ValueError("input genes for hidden nodes have invalid value")

        for region_idx, output_region in self.iter_output_regions(dna):

            if output_region[0] != self._id_output_node:
                raise ValueError(
                    "function genes for output nodes need to be"
                    "identical to output node identifiers"
                )

            if output_region[1] not in self._permissible_inputs_for_output_region():
                raise ValueError("input gene for output nodes has invalid value")

            if output_region[2:] != [self._id_unused_gene] * (self._primitives.max_arity - 1):
                raise ValueError("inactive input genes for output nodes need to be empty")

    def _hidden_column_idx(self, region_idx: int) -> int:
        assert self._n_inputs <= region_idx
        assert region_idx < self._n_inputs + self._n_rows * self._n_columns
        hidden_column_idx = (region_idx - self._n_inputs) // self._n_rows
        assert 0 <= hidden_column_idx
        assert hidden_column_idx < self._n_columns
        return hidden_column_idx

    def iter_input_regions(
        self, dna: Optional[List[int]] = None
    ) -> Generator[Tuple[int, list], None, None]:
        if dna is None:
            dna = self.dna
        for i in range(self._n_inputs):
            region_idx = i
            region = dna[
                region_idx * self._length_per_region : (region_idx + 1) * self._length_per_region
            ]
            yield region_idx, region

    def iter_hidden_regions(
        self, dna: Optional[List[int]] = None
    ) -> Generator[Tuple[int, List[int]], None, None]:
        if dna is None:
            dna = self.dna
        for i in range(self._n_hidden):
            region_idx = i + self._n_inputs
            region = dna[
                region_idx * self._length_per_region : (region_idx + 1) * self._length_per_region
            ]
            yield region_idx, region

    def iter_output_regions(
        self, dna: Optional[List[int]] = None
    ) -> Generator[Tuple[int, List[int]], None, None]:
        if dna is None:
            dna = self.dna
        for i in range(self._n_outputs):
            region_idx = i + self._n_inputs + self._n_hidden
            region = dna[
                region_idx * self._length_per_region : (region_idx + 1) * self._length_per_region
            ]
            yield region_idx, region

    def _is_gene_in_input_region(self, gene_idx: int) -> bool:
        return gene_idx < (self._n_inputs * self._length_per_region)

    def _is_gene_in_hidden_region(self, gene_idx: int) -> bool:
        return ((self._n_inputs * self._length_per_region) <= gene_idx) & (
            gene_idx < ((self._n_inputs + self._n_hidden) * self._length_per_region)
        )

    def _is_gene_in_output_region(self, gene_idx: int) -> bool:
        return ((self._n_inputs + self._n_hidden) * self._length_per_region) <= gene_idx

    def _is_input_region(self, region_idx: int) -> bool:
        return region_idx < self._n_inputs

    def _is_hidden_region(self, region_idx: int) -> bool:
        return (self._n_inputs <= region_idx) & (region_idx < self._n_inputs + self._n_hidden)

    def _is_output_region(self, region_idx: int) -> bool:
        return self._n_inputs + self._n_hidden <= region_idx

    def _is_function_gene(self, gene_idx: int) -> bool:
        return (gene_idx % self._length_per_region) == 0

    def _is_active_input_gene(self, gene_idx: int) -> bool:
        input_index = gene_idx % self._length_per_region
        assert input_index > 0
        region_idx = gene_idx // self._length_per_region
        if self._is_input_region(region_idx):
            return False
        elif self._is_hidden_region(region_idx):
            node_arity = self._primitives[self._dna[region_idx * self._length_per_region]]._arity
            return input_index <= node_arity
        elif self._is_output_region(region_idx):
            return input_index == 1
        else:
            assert False  # should never be reached

    def mutate(self, n_mutations: int, active_regions: List[int], rng: np.random.RandomState):
        """Mutate the genome.

        Parameters
        ----------
        n_mutations : int
            Number of entries in the genome to be mutated.
        active_regions: List[int]
            Regions in the genome that are currently used in the
            computational graph. Used to check whether mutations are
            silent or require reevaluation of fitness.
        rng : numpy.random.RandomState
            Random number generator instance to use for crossover.

        Returns
        ----------
        True if only inactive regions of the genome were mutated, False otherwise.
        """
        assert isinstance(n_mutations, int) and 0 < n_mutations

        successful_mutations = 0
        only_silent_mutations = True
        while successful_mutations < n_mutations:

            gene_idx = rng.randint(0, self._n_genes)
            region_idx = gene_idx // self._length_per_region

            if self._is_input_region(region_idx):
                continue  # nothing to do here

            elif self._is_output_region(region_idx):
                success = self._mutate_output_region(gene_idx, rng)
                if success:
                    silent = False
                    only_silent_mutations = only_silent_mutations and silent
                    successful_mutations += 1

            elif self._is_hidden_region(region_idx):
                silent = self._mutate_hidden_region(gene_idx, active_regions, rng)
                only_silent_mutations = only_silent_mutations and silent

            else:
                assert False  # should never be reached

        self._validate_dna(self._dna)

        return only_silent_mutations

    def _mutate_output_region(self, gene_idx, rng):
        assert self._is_gene_in_output_region(gene_idx)

        if not self._is_function_gene(gene_idx) and self._is_active_input_gene(gene_idx):
            permissible_inputs = self._permissible_inputs_for_output_region()
            self._dna[gene_idx] = rng.choice(permissible_inputs)
            return True
        else:
            return False

    def _mutate_hidden_region(
        self, gene_idx: int, active_regions: List[int], rng: np.random.RandomState
    ) -> bool:

        assert self._is_gene_in_hidden_region(gene_idx)

        region_idx = gene_idx // self._length_per_region
        silent_mutation = region_idx not in active_regions

        if self._is_function_gene(gene_idx):
            self._dna[gene_idx] = self._primitives.sample_allele(rng)
            return silent_mutation

        else:
            permissible_inputs = self._permissible_inputs(region_idx)
            self._dna[gene_idx] = rng.choice(permissible_inputs)

            silent_mutation = silent_mutation or (not self._is_active_input_gene(gene_idx))
            return silent_mutation

    @property
    def primitives(self) -> Primitives:
        return self._primitives

    def clone(self) -> "Genome":
        """Clone the genome.

        Returns
        -------
        Genome
        """

        new = Genome(
            self._n_inputs,
            self._n_outputs,
            self._n_columns,
            self._n_rows,
            self._levels_back,
            tuple(self._primitives),
        )
        new.dna = self._dna.copy()

        # Lamarckian strategy: parameter values are passed on to
        # offspring
        new.parameter_names_to_values = self.parameter_names_to_values.copy()

        return new

    def update_parameters_from_torch_class(self, torch_cls: "torch.nn.Module") -> bool:
        """Update values stored in Parameter nodes of graph from parameters of
        a given Torch instance.  Can be used to import new values from
        a Torch class after they have been altered, e.g., by local
        search.

        Parameters
        ----------
        torch_cls : torch.nn.module
            Instance of a torch class.

        Returns
        -------
        bool
            Whether any parameter was updated

        """
        any_parameter_updated = False

        for name, value in torch_cls.named_parameters():
            name = f"<{name[1:]}>"
            if name in self.parameter_names_to_values:
                self.parameter_names_to_values[name] = value.item()
                assert not np.isnan(self.parameter_names_to_values[name])
                any_parameter_updated = True

        return any_parameter_updated
