import re
from typing import Dict, Generator, List, Optional, Set, Tuple, Type

import numpy as np

from .cartesian_graph import CartesianGraph
from .node import Node, OperatorNode
from .primitives import Primitives

try:
    import torch  # noqa: F401

    torch_available = True
except ModuleNotFoundError:
    torch_available = False


ID_INPUT_NODE: int = -1
ID_OUTPUT_NODE: int = -2
ID_NON_CODING_GENE: int = -3


class Genome:
    """Genome class for  individuals.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_columns: int,
        n_rows: int,
        primitives: Tuple[Type[Node], ...],
        levels_back: Optional[int] = None,
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
        primitives : Tuple[Type[Node], ...]
           Tuple of primitives that the genome can refer to.
        levels_back : Optional[int]
            Maximal column distance of inputs to an internal node. If
            set to `None`, no restrictions are used. Defaults to None

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
        if levels_back == 0 and n_columns != 0:
            raise ValueError("levels_back must be strictly positive")
        if levels_back > n_columns:
            raise ValueError("levels_back can not be larger than n_columns")
        self._levels_back = levels_back

        self._primitives = Primitives(primitives)
        self._length_per_region = (
            1 + self._primitives.max_arity
        )  # one function gene + multiple address genes

        self._dna: List[int] = []  # stores dna as list of alleles for all regions

        # constants used as identifiers for input and output nodes
        self._id_input_node: int = ID_INPUT_NODE
        self._id_output_node: int = ID_OUTPUT_NODE
        self._id_unused_gene: int = ID_NON_CODING_GENE

        # dictionary to store values of Parameter nodes
        self._parameter_names_to_values: Dict[str, float] = {}

        # list of permissible values for every gene
        self._permissible_values: List[np.ndarray] = self.determine_permissible_values()

    def __getitem__(self, key: int) -> int:
        if self.dna is None:
            raise RuntimeError("dna not initialized")
        return self.dna[key]

    def __setitem__(self, key: int, value: int) -> None:
        dna = list(self._dna)
        dna[key] = value
        self.dna = dna

    @property
    def dna(self) -> List[int]:
        return list(self._dna)  # return copy to avoid inplace modification

    @dna.setter
    def dna(self, value: List[int]) -> None:
        self._validate_dna(value)
        self._dna = value
        self._initialize_unknown_parameters()

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

    def determine_permissible_values_per_gene(self, gene_idx: int) -> np.ndarray:
        region_idx = self._get_region_idx(gene_idx)

        if self._is_input_region(region_idx):
            return self._determine_permissible_values_input_region(gene_idx)

        elif self._is_hidden_region(region_idx):
            return self._determine_permissible_values_hidden_region(gene_idx, region_idx)

        elif self._is_output_region(region_idx):
            return self._determine_permissible_values_output_region(gene_idx)

        else:
            assert False  # should never be reached

    def determine_permissible_values(self) -> List[np.ndarray]:
        """Determine permissible values for every gene.

        Parameters
        ----------
        None

        Returns
        ----------
        permissible_values
            List[numpy.ndarray]: List of permissible values for every gene
        """
        permissible_values: List[np.ndarray] = []
        for gene_idx in range(self._n_genes):
            permissible_values_per_gene = self.determine_permissible_values_per_gene(gene_idx)
            permissible_values.append(permissible_values_per_gene)
        return permissible_values

    def _determine_permissible_values_input_region(self, gene_idx: int) -> np.ndarray:

        if self._is_function_gene(gene_idx):
            return np.array(self._id_input_node)
        else:
            return np.array(self._id_unused_gene)

    def _determine_permissible_values_hidden_region(
        self, gene_idx: int, region_idx: int
    ) -> np.ndarray:

        if self._is_function_gene(gene_idx):
            return np.arange(len(self._primitives._primitives))

        elif self._is_hidden_address_gene(gene_idx, region_idx):
            return np.array(self._permissible_addresses(region_idx))

        else:
            assert False  # should never be reached

    def _determine_permissible_values_output_region(self, gene_idx: int) -> np.ndarray:

        if self._is_function_gene(gene_idx):
            return np.array(self._id_output_node)
        else:
            address_idx = gene_idx % self._length_per_region - 1
            if address_idx == 0:
                return np.array(self._permissible_addresses_for_output_region())
            else:
                return np.array(self._id_unused_gene)

    def _create_input_region(self) -> List[int]:

        region = []
        region.append(self._id_input_node)
        # non-coding address genes since input nodes do not receive
        # inputs themselves
        region += [self._id_unused_gene] * self._primitives.max_arity
        return region

    def _create_random_hidden_region(
        self, rng: np.random.RandomState, permissible_addresses: List[int]
    ) -> List[int]:

        region = []
        node_id = self._primitives.sample_allele(rng)
        region.append(node_id)
        region += list(rng.choice(permissible_addresses, self._primitives.max_arity))

        return region

    def _create_random_output_region(
        self, rng: np.random.RandomState, permissible_addresses: List[int]
    ) -> List[int]:

        region = []
        region.append(self._id_output_node)
        region.append(rng.choice(permissible_addresses))
        # output nodes have only one address, other genes are hence non-coding
        region += [self._id_unused_gene] * (self._primitives.max_arity - 1)

        return region

    def randomize(self, rng: np.random.RandomState) -> None:
        """Randomize the genome.

        Parameters
        ----------
        rng : numpy.RandomState
            Random number generator instance to use for randomizing.

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

            if i % self._n_rows == 0:  # only compute permissible addresses once per column
                permissible_addresses = self._permissible_addresses(i + self._n_inputs)

            dna += self._create_random_hidden_region(rng, permissible_addresses)

        # add output nodes
        permissible_addresses = (
            self._permissible_addresses_for_output_region()
        )  # identical for outputs
        for i in range(self._n_outputs):
            dna += self._create_random_output_region(rng, permissible_addresses)

        # accept generated dna if it is valid
        self.dna = dna

    def reorder(self, rng: np.random.RandomState) -> None:
        """Reorder the genome

        Shuffle node ordering of internal (hidden) nodes in genome without changing the phenotype.
        (Goldman 2015, DOI: 10.1109/TEVC.2014.2324539)

        During reordering, inactive genes, e.g., address genes of nodes with arity zero,
        are not taken into account and can hence have invalid values after reordering.
        These invalid values are replaced by random values
        for the respective gene after reordering.

        Parameters
        ----------
        rng : numpy.RandomState
            Random number generator instance.

        Returns
        ----------
        None
        """
        if (self._n_rows != 1) or (self._levels_back != self._n_columns):
            raise ValueError(
                "Genome reordering is only implemented for n_rows=1" " and levels_back=n_columns"
            )

        dna = self._dna.copy()

        node_dependencies: Dict[int, Set[int]] = self._determine_node_dependencies()

        addable_nodes: Set[int] = self._get_addable_nodes(node_dependencies)

        new_node_idx: int = self._n_inputs  # First position to be placed is after inputs
        used_node_indices: List[int] = []
        old_to_new_parameter_names_to_values: Dict[Tuple[str, str], float] = {}

        while len(addable_nodes) > 0:

            old_node_idx = rng.choice(list(addable_nodes))
            dna = self._copy_dna_segment(dna, old_node_idx=old_node_idx, new_node_idx=new_node_idx)

            old_to_new_parameter_names_to_values.update(
                self._convert_parameter_names(old_node_idx, new_node_idx)
            )

            for dependencies in node_dependencies.values():
                dependencies.discard(old_node_idx)

            used_node_indices.append(old_node_idx)
            addable_nodes = self._get_addable_nodes(node_dependencies, used_node_indices)
            new_node_idx += 1

        self._update_address_genes(dna, used_node_indices)
        self._replace_invalid_address_alleles(dna, rng)
        self._update_parameters_names_to_values(old_to_new_parameter_names_to_values)

        self.dna = dna

    def _convert_parameter_names(
        self, old_node_idx: int, new_node_idx: int
    ) -> Dict[Tuple[str, str], float]:
        node_id: int = self.dna[old_node_idx * self._length_per_region]
        node_type = self._primitives[node_id]
        d: Dict[Tuple[str, str], float] = {}
        if issubclass(node_type, OperatorNode):
            for old_parameter_name in self._get_parameter_names_with_idx_of_node(
                node_type, old_node_idx
            ):
                g = re.findall(f"<([a-z]+){old_node_idx}>", old_parameter_name)
                if len(g) != 0:
                    assert len(g) == 1
                    new_parameter_name: str = "<" + g[0] + str(new_node_idx) + ">"
                    d[(old_parameter_name, new_parameter_name)] = self._parameter_names_to_values[
                        old_parameter_name
                    ]
        return d

    def _update_parameters_names_to_values(
        self, old_to_new_parameter_names_to_values: Dict[Tuple[str, str], float]
    ) -> None:
        # first we delete all old parameter names
        for old_parameter_name, _ in old_to_new_parameter_names_to_values:
            del self._parameter_names_to_values[old_parameter_name]

        # then we add new parameter names and corresponding values
        for (_, new_parameter_name), v in old_to_new_parameter_names_to_values.items():
            self._parameter_names_to_values[new_parameter_name] = v

    def _copy_dna_segment(self, dna: List[int], old_node_idx: int, new_node_idx: int) -> List[int]:
        """ Copy a nodes dna segment from its old node location to a new location. """

        dna[
            new_node_idx * self._length_per_region : (new_node_idx + 1) * self._length_per_region
        ] = self._dna[
            old_node_idx * self._length_per_region : (old_node_idx + 1) * self._length_per_region
        ]

        return dna

    def _update_address_genes(self, dna: List[int], used_node_indices: List[int]) -> None:
        """Update address genes of all nodes from old node indices to new node indices"""
        for gene_idx, gene_value in enumerate(dna):
            region_idx = self._get_region_idx(gene_idx)
            if self._is_hidden_address_gene(gene_idx, region_idx) or self._is_output_address_gene(
                gene_idx
            ):
                if gene_value >= self._n_inputs:
                    gene_value = self._n_inputs + used_node_indices.index(gene_value)
            dna[gene_idx] = gene_value

    def _replace_invalid_address_alleles(self, dna: List[int], rng: np.random.RandomState) -> None:
        """Replace invalid alleles for unused address genes of all nodes
        by random permissible values.
        WARNING: Works only if self.n_rows==1.
        """
        assert self._n_rows == 1

        for gene_idx, gene_value in enumerate(dna):
            region_idx = self._get_region_idx(gene_idx)
            if self._is_hidden_address_gene(gene_idx, region_idx) and gene_value > region_idx:
                permissible_values = self.determine_permissible_values_per_gene(gene_idx)
                gene_value = rng.choice(permissible_values)
                dna[gene_idx] = gene_value

    def _get_addable_nodes(
        self, node_dependencies: Dict[int, Set[int]], used_node_indices: List[int] = [],
    ) -> Set[int]:
        """ Get the set of addable nodes,
         nodes which have no dependencies and were not already used.
        """
        addable_nodes = set(
            idx for idx, dependencies in node_dependencies.items() if len(dependencies) == 0
        )
        return addable_nodes.difference(used_node_indices)

    def _get_region_idx(self, gene_idx: int) -> int:
        return gene_idx // self._length_per_region

    def _get_region(self, region_idx: int, dna: List[int]) -> List[int]:
        return dna[
            region_idx * self._length_per_region : (region_idx + 1) * self._length_per_region
        ]

    def _determine_node_dependencies(self) -> Dict[int, Set[int]]:
        """ Determines the set of node indices on which each node depends.
            Unused address genes are ignored.

        Returns
        ----
        dependencies: Dict[int, Set[int]]
            Dictionary containing for every node the set of active address genes

        """
        dependencies: Dict[int, Set[int]] = {}
        for region_idx, _ in self.iter_hidden_regions():

            current_node_dependencies: Set[int] = set()

            operator_idx: int = region_idx * self._length_per_region

            current_arity: int = self._determine_operator_arity(operator_idx)

            for idx_gene in range(
                1, current_arity + 1
            ):  # shift by 1 since first gene is the operator gene
                address = self._dna[operator_idx + idx_gene]
                if not self._is_input_region(
                    address
                ):  # not necessary to add input regions, since their positions remain fixed
                    current_node_dependencies.add(address)

            dependencies[region_idx] = current_node_dependencies

        return dependencies

    def _determine_operator_arity(self, gene_idx: int) -> int:

        assert self._is_function_gene(gene_idx)

        return self._primitives[self._dna[gene_idx]]._arity

    def _permissible_addresses(self, region_idx: int) -> List[int]:

        assert not self._is_input_region(region_idx)

        permissible_addresses = []

        # all nodes can be connected to input
        permissible_addresses += [j for j in range(0, self._n_inputs)]

        # add all nodes reachable according to levels back
        if self._is_hidden_region(region_idx):
            hidden_column_idx = self._hidden_column_idx(region_idx)
            lower = self._n_inputs + self._n_rows * max(0, (hidden_column_idx - self._levels_back))
            upper = self._n_inputs + self._n_rows * hidden_column_idx
        else:
            assert self._is_output_region(region_idx)
            lower = self._n_inputs
            upper = self._n_inputs + self._n_rows * self._n_columns

        permissible_addresses += [j for j in range(lower, upper)]

        return permissible_addresses

    def _permissible_addresses_for_output_region(self) -> List[int]:
        return self._permissible_addresses(self._n_inputs + self._n_rows * self._n_columns)

    def _validate_dna(self, dna: List[int]) -> None:

        if len(dna) != self._n_genes:
            raise ValueError("dna length mismatch")

        for region_idx, input_region in self.iter_input_regions(dna):

            if input_region[0] != self._id_input_node:
                raise ValueError(
                    "function genes for input nodes need to be identical to input node identifiers"
                )

            if input_region[1:] != ([self._id_unused_gene] * self._primitives.max_arity):
                raise ValueError("address genes for input nodes need to be empty")

        for region_idx, hidden_region in self.iter_hidden_regions(dna):

            if not self._primitives.is_valid_allele(hidden_region[0]):
                raise ValueError("function gene for hidden node has invalid value")

            address_genes = hidden_region[1:]

            permissible_addresses = set(self._permissible_addresses(region_idx))
            if not set(address_genes).issubset(permissible_addresses):
                raise ValueError("address genes for hidden nodes have invalid value")

        for region_idx, output_region in self.iter_output_regions(dna):

            if output_region[0] != self._id_output_node:
                raise ValueError(
                    "function genes for output nodes need to be"
                    "identical to output node identifiers"
                )

            if output_region[1] not in self._permissible_addresses_for_output_region():
                raise ValueError("address gene for output nodes has invalid value")

            if output_region[2:] != [self._id_unused_gene] * (self._primitives.max_arity - 1):
                raise ValueError("inactive address genes for output nodes need to be empty")

    def _hidden_column_idx(self, region_idx: int) -> int:
        assert self._n_inputs <= region_idx
        assert region_idx < self._n_inputs + self._n_rows * self._n_columns
        hidden_column_idx = (region_idx - self._n_inputs) // self._n_rows
        assert 0 <= hidden_column_idx
        assert hidden_column_idx < self._n_columns
        return hidden_column_idx

    def iter_input_regions(
        self, dna: Optional[List[int]] = None
    ) -> Generator[Tuple[int, List[int]], None, None]:
        if dna is None:
            dna = self.dna
        for i in range(self._n_inputs):
            region_idx = i
            region = self._get_region(region_idx, dna)
            yield region_idx, region

    def iter_hidden_regions(
        self, dna: Optional[List[int]] = None
    ) -> Generator[Tuple[int, List[int]], None, None]:
        if dna is None:
            dna = self.dna
        for i in range(self._n_hidden):
            region_idx = i + self._n_inputs
            region = self._get_region(region_idx, dna)
            yield region_idx, region

    def iter_output_regions(
        self, dna: Optional[List[int]] = None
    ) -> Generator[Tuple[int, List[int]], None, None]:
        if dna is None:
            dna = self.dna
        for i in range(self._n_outputs):
            region_idx = i + self._n_inputs + self._n_hidden
            region = self._get_region(region_idx, dna)
            yield region_idx, region

    def _is_gene_in_input_region(self, gene_idx: int) -> bool:
        return gene_idx < (self._n_inputs * self._length_per_region)

    def _is_gene_in_hidden_region(self, gene_idx: int) -> bool:
        return ((self._n_inputs * self._length_per_region) <= gene_idx) and (
            gene_idx < ((self._n_inputs + self._n_hidden) * self._length_per_region)
        )

    def _is_gene_in_output_region(self, gene_idx: int) -> bool:
        return ((self._n_inputs + self._n_hidden) * self._length_per_region) <= gene_idx

    def _is_input_region(self, region_idx: int) -> bool:
        return region_idx < self._n_inputs

    def _is_hidden_region(self, region_idx: int) -> bool:
        return (self._n_inputs <= region_idx) and (region_idx < self._n_inputs + self._n_hidden)

    def _is_output_region(self, region_idx: int) -> bool:
        return self._n_inputs + self._n_hidden <= region_idx

    def _is_function_gene(self, gene_idx: int) -> bool:
        return (gene_idx % self._length_per_region) == 0

    def _is_hidden_address_gene(self, gene_idx: int, region_idx: int) -> bool:
        return self._is_hidden_region(region_idx) and (not self._is_function_gene(gene_idx))

    def _is_output_address_gene(self, gene_idx: int) -> bool:
        return (
            self._is_gene_in_output_region(gene_idx) and gene_idx % self._length_per_region == 1
        )  # assumes 2nd gene in output is coding for address

    def _select_gene_indices_for_mutation(
        self, mutation_rate: float, rng: np.random.RandomState
    ) -> List[int]:
        """Selects the gene indices for mutations

        Parameters
        ----------
        mutation_rate : float
            Probability of a gene to be mutated, between 0 (excluded) and 1 (included).
        rng : numpy.random.RandomState
            Random number generator instance to use for selecting the indices.

        Returns
        ----------
        selected_gene_indices: np.ndarray
            indices of the genes selected for mutation.
        """

        selected_gene_indices = np.nonzero(rng.rand(len(self.dna)) < mutation_rate)[0]
        return selected_gene_indices

    def mutate(self, mutation_rate: float, rng: np.random.RandomState):
        """Mutate the genome.

        Parameters
        ----------
        mutation_rate : float
            Probability of a gene to be mutated, between 0 (excluded) and 1 (included).
        rng : numpy.random.RandomState
            Random number generator instance to use for mutating.


        Returns
        ----------
        bool
            True if only inactive regions of the genome were mutated, False otherwise.
        """

        graph = CartesianGraph(self)
        active_regions = graph.determine_active_regions()
        dna = list(self._dna)
        only_silent_mutations = True

        selected_gene_indices = self._select_gene_indices_for_mutation(mutation_rate, rng)

        for (gene_idx, allele) in zip(selected_gene_indices, np.array(dna)[selected_gene_indices]):

            region_idx = self._get_region_idx(gene_idx)

            permissible_values = self._permissible_values[gene_idx]
            permissible_alternative_values = permissible_values[permissible_values != allele]

            if len(permissible_alternative_values) > 0:

                dna[gene_idx] = rng.choice(permissible_alternative_values)
                modified_parameter_value: bool
                if self._is_function_gene(gene_idx):
                    region_idx = self._get_region_idx(gene_idx)
                    region = self._get_region(region_idx, dna)
                    modified_parameter_value = self._initialize_parameter_values(
                        region_idx, region, reinitialize=True
                    )
                else:
                    modified_parameter_value = False

                silent = (region_idx not in active_regions) and (not modified_parameter_value)
                only_silent_mutations = only_silent_mutations and silent

        self.dna = dna
        return only_silent_mutations

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
            tuple(self._primitives),
            self._levels_back,
        )
        new.dna = self.dna.copy()

        # Lamarckian strategy: parameter values are passed on to
        # offspring
        new._parameter_names_to_values = self._parameter_names_to_values.copy()

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
            if name in self._parameter_names_to_values:
                self._parameter_names_to_values[name] = value.item()
                assert not np.isnan(self._parameter_names_to_values[name])
                any_parameter_updated = True

        return any_parameter_updated

    def _initialize_unknown_parameters(self) -> None:
        for region_idx, region in self.iter_hidden_regions():
            node_id = region[0]
            node_type = self._primitives[node_id]
            assert issubclass(node_type, OperatorNode)
            for parameter_name_with_idx in self._get_parameter_names_with_idx_of_node(
                node_type, region_idx
            ):
                if parameter_name_with_idx not in self._parameter_names_to_values:
                    self._parameter_names_to_values[
                        parameter_name_with_idx
                    ] = node_type.initial_value(parameter_name_with_idx)

    def _get_parameter_names_with_idx_of_node(
        self, node_type: Type[OperatorNode], region_idx: int
    ) -> List[str]:
        parameter_names_with_idx: List[str] = []
        for parameter_name in node_type._parameter_names:
            parameter_names_with_idx.append("<" + parameter_name[1:-1] + str(region_idx) + ">")
        return parameter_names_with_idx

    def _initialize_parameter_values(
        self, region_idx: int, region: List[int], reinitialize: bool = False
    ) -> bool:
        node_id = region[0]
        node_type = self._primitives[node_id]
        assert issubclass(node_type, OperatorNode)
        modified_parameter_value: bool = False
        for parameter_name in node_type._parameter_names:
            parameter_name_with_idx = "<" + parameter_name[1:-1] + str(region_idx) + ">"
            if reinitialize or (parameter_name_with_idx not in self._parameter_names_to_values):
                self._parameter_names_to_values[parameter_name_with_idx] = node_type.initial_value(
                    parameter_name_with_idx
                )
                modified_parameter_value = True
        return modified_parameter_value

    def parameters_to_numpy_array(self, only_active_nodes: bool = False) -> "np.ndarray[float]":
        if only_active_nodes:
            graph = CartesianGraph(self)
            active_regions: List[int] = graph.determine_active_regions()
            params_names: List[str] = []
            params: List[float] = []
            for p in self._parameter_names_to_values:
                region_idx: int = self._region_idx_from_parameter_name(p)
                if region_idx in active_regions:
                    params_names.append(p)
                    params.append(self._parameter_names_to_values[p])
            return np.fromiter(params, dtype=np.float), params_names
        else:
            return (
                np.fromiter(self._parameter_names_to_values.values(), dtype=np.float),
                list(self._parameter_names_to_values.keys()),
            )

    def _region_idx_from_parameter_name(self, parameter_name: str) -> int:
        return int(re.findall("<[A-z]+([0-9]+)>", parameter_name)[0])

    def update_parameters_from_numpy_array(
        self, params: "np.ndarray[float]", params_names: List[str]
    ) -> bool:
        any_parameter_updated: bool = False
        for v, p in zip(params, params_names):
            assert p in self._parameter_names_to_values
            if not np.isclose(self._parameter_names_to_values[p], v, rtol=0.0):
                self._parameter_names_to_values[p] = v
                any_parameter_updated = True
        return any_parameter_updated
