import math

import numpy as np
import pytest

import cgp
from cgp.cartesian_graph import CartesianGraph
from cgp.genome import ID_INPUT_NODE, ID_NON_CODING_GENE, ID_OUTPUT_NODE


def test_check_dna_consistency():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"], params["n_outputs"], params["n_columns"], params["n_rows"], primitives,
    )
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        0,
        ID_NON_CODING_GENE,
    ]

    # invalid length
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            ID_OUTPUT_NODE,
            ID_INPUT_NODE,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
            0,
        ]

    # invalid function gene for input node
    with pytest.raises(ValueError):
        genome.dna = [
            0,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            ID_OUTPUT_NODE,
            0,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid address gene for input node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            0,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            ID_OUTPUT_NODE,
            0,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid function gene for hidden node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            2,
            0,
            1,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid address gene for hidden node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            2,
            1,
            ID_OUTPUT_NODE,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid function gene for output node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            0,
            1,
            0,
            0,
            ID_NON_CODING_GENE,
        ]

    # invalid address gene for input node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            0,
            1,
            ID_OUTPUT_NODE,
            3,
            ID_NON_CODING_GENE,
        ]

    # invalid inactive address gene for output node
    with pytest.raises(ValueError):
        genome.dna = [
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            ID_INPUT_NODE,
            ID_NON_CODING_GENE,
            ID_NON_CODING_GENE,
            0,
            0,
            1,
            ID_OUTPUT_NODE,
            0,
            0,
        ]


def test_permissible_addresses():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 4, "n_rows": 3, "levels_back": 2}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        primitives,
        params["levels_back"],
    )
    genome.randomize(np.random)

    for input_idx in range(params["n_inputs"]):
        region_idx = input_idx
        with pytest.raises(AssertionError):
            genome._permissible_addresses(region_idx)

    expected_for_hidden = [
        [0, 1],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 5, 6, 7, 8, 9, 10],
    ]

    for column_idx in range(params["n_columns"]):
        region_idx = params["n_inputs"] + params["n_rows"] * column_idx
        assert expected_for_hidden[column_idx] == genome._permissible_addresses(region_idx)

    expected_for_output = list(range(params["n_inputs"] + params["n_rows"] * params["n_columns"]))

    for output_idx in range(params["n_outputs"]):
        region_idx = params["n_inputs"] + params["n_rows"] * params["n_columns"] + output_idx
        assert expected_for_output == genome._permissible_addresses(region_idx)


def test_region_iterators():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"], params["n_outputs"], params["n_columns"], params["n_rows"], primitives,
    )
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        0,
        ID_NON_CODING_GENE,
    ]

    for region_idx, region in genome.iter_input_regions():
        assert region == [ID_INPUT_NODE, ID_NON_CODING_GENE, ID_NON_CODING_GENE]

    for region_idx, region in genome.iter_hidden_regions():
        assert region == [0, 0, 1]

    for region_idx, region in genome.iter_output_regions():
        assert region == [ID_OUTPUT_NODE, 0, ID_NON_CODING_GENE]


def test_check_levels_back_consistency():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 4, "n_rows": 3, "levels_back": None}

    primitives = (cgp.Add,)

    params["levels_back"] = 0
    with pytest.raises(ValueError):
        cgp.Genome(
            params["n_inputs"],
            params["n_outputs"],
            params["n_columns"],
            params["n_rows"],
            primitives,
            params["levels_back"],
        )

    params["levels_back"] = params["n_columns"] + 1
    with pytest.raises(ValueError):
        cgp.Genome(
            params["n_inputs"],
            params["n_outputs"],
            params["n_columns"],
            params["n_rows"],
            primitives,
            params["levels_back"],
        )

    params["levels_back"] = params["n_columns"] - 1
    cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        primitives,
        params["levels_back"],
    )


def test_catch_invalid_allele_in_inactive_region():
    primitives = (cgp.ConstantFloat,)
    genome = cgp.Genome(1, 1, 1, 1, primitives)

    # should raise error: ConstantFloat node has no addresses, but silent
    # address gene should still specify valid address
    with pytest.raises(ValueError):
        genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, ID_NON_CODING_GENE, ID_OUTPUT_NODE, 1]

    # correct
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]


def test_individuals_have_different_genome(population_params, genome_params, ea_params):
    def objective(ind):
        ind.fitness = 1.0
        return ind

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    pop._generate_random_parent_population()

    ea.initialize_fitness_parents(pop, objective)

    ea.step(pop, objective)

    for i, parent_i in enumerate(pop._parents):
        for j, parent_j in enumerate(pop._parents):
            if i != j:
                assert parent_i is not parent_j
                assert parent_i.genome is not parent_j.genome
                assert parent_i.genome.dna is not parent_j.genome.dna


def test_is_gene_in_input_region(rng_seed):
    genome = cgp.Genome(2, 1, 2, 1, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_input_region(0)
    assert not genome._is_gene_in_input_region(6)


def test_is_gene_in_hidden_region(rng_seed):
    genome = cgp.Genome(2, 1, 2, 1, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_hidden_region(6)
    assert genome._is_gene_in_hidden_region(9)
    assert not genome._is_gene_in_hidden_region(5)
    assert not genome._is_gene_in_hidden_region(12)


def test_is_gene_in_output_region(rng_seed):
    genome = cgp.Genome(2, 1, 2, 1, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_output_region(12)
    assert not genome._is_gene_in_output_region(11)


def test_mutation_rate(rng_seed, mutation_rate):
    n_inputs = 1
    n_outputs = 1
    n_columns = 4
    n_rows = 3
    genome = cgp.Genome(n_inputs, n_outputs, n_columns, n_rows, (cgp.Add, cgp.Sub))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    def count_n_immutable_genes(n_inputs, n_outputs, n_rows):
        length_per_region = genome.primitives.max_arity + 1  # function gene + address gene
        n_immutable_genes = n_inputs * length_per_region  # none of the input genes are mutable
        n_immutable_genes += n_outputs * (
            length_per_region - 1
        )  # only one gene per output can be mutated
        if n_inputs == 1:
            n_immutable_genes += (
                n_rows * genome.primitives.max_arity
            )  # address gene in the first (hidden) column can't be mutated
            # if only one input node exists
        return n_immutable_genes

    def count_mutations(dna0, dna1):
        n_differences = 0
        for (allele0, allele1) in zip(dna0, dna1):
            if allele0 != allele1:
                n_differences += 1
        return n_differences

    n_immutable_genes = count_n_immutable_genes(n_inputs, n_outputs, n_rows)
    n_mutations_mean_expected = mutation_rate * (len(genome.dna) - n_immutable_genes)
    n_mutations_std_expected = np.sqrt(
        (len(genome.dna) - n_immutable_genes) * mutation_rate * (1 - mutation_rate)
    )

    n = 10000
    n_mutations = []
    for _ in range(n):
        dna_old = genome.dna
        genome.mutate(mutation_rate, rng)
        n_mutations.append(count_mutations(dna_old, genome.dna))

    assert np.mean(n_mutations) == pytest.approx(n_mutations_mean_expected, rel=0.04)
    assert np.std(n_mutations) == pytest.approx(n_mutations_std_expected, rel=0.04)


def test_only_silent_mutations(genome_params, mutation_rate, rng_seed):
    genome = cgp.Genome(**genome_params)
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    only_silent_mutations = genome.mutate(mutation_rate=0, rng=rng)
    assert only_silent_mutations is True

    only_silent_mutations = genome.mutate(mutation_rate=1, rng=rng)
    assert not only_silent_mutations

    dna_fixed = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        2,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    genome.dna = dna_fixed
    graph = CartesianGraph(genome)
    active_regions = graph.determine_active_regions()
    length_per_region = genome.primitives.max_arity + 1  # function gene + address gene
    gene_to_be_mutated_non_active = (
        3 * length_per_region
    )  # operator gene of 2nd hidden node, a silent node in this dna configuration

    def select_gene_indices_silent(mutation_rate, rng):
        selected_gene_indices = [gene_to_be_mutated_non_active]
        return selected_gene_indices

    genome._select_gene_indices_for_mutation = (
        select_gene_indices_silent  # monkey patch the selection of indices to select a silent gene
    )
    genome.dna = dna_fixed
    only_silent_mutations = genome.mutate(mutation_rate, rng)
    assert only_silent_mutations is True

    gene_to_be_mutated_active = (
        active_regions[-1] * length_per_region
    )  # operator gene of the 1st active hidden node,
    # should always be mutable and result in a non-silent-mutation

    def select_gene_indices_non_silent(mutation_rate, rng):
        selected_gene_indices = [gene_to_be_mutated_active]
        return selected_gene_indices

    genome._select_gene_indices_for_mutation = select_gene_indices_non_silent
    # monkey patch the selection of indices to select a non-silent gene
    only_silent_mutations = genome.mutate(mutation_rate, rng)
    assert not only_silent_mutations


def test_permissible_values(genome_params):
    n_inputs = 2
    n_outputs = 1
    n_columns = 3
    n_rows = 3
    levels_back = 2
    primitives = (cgp.Add, cgp.Sub, cgp.ConstantFloat)
    genome = cgp.Genome(n_inputs, n_outputs, n_columns, n_rows, primitives, levels_back)

    permissible_function_gene_values = np.arange(len(genome._primitives._primitives))
    permissible_addresses_first_internal_column = np.array([0, 1])
    permissible_addresses_second_internal_column = np.array([0, 1, 2, 3, 4])
    permissible_addresses_third_internal_column = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    permissible_addresses_output = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    permissible_values_expected = [
        np.array(ID_INPUT_NODE),
        np.array(ID_NON_CODING_GENE),
        np.array(ID_NON_CODING_GENE),
        np.array(ID_INPUT_NODE),
        np.array(ID_NON_CODING_GENE),
        np.array(ID_NON_CODING_GENE),
        permissible_function_gene_values,
        permissible_addresses_first_internal_column,
        permissible_addresses_first_internal_column,
        permissible_function_gene_values,
        permissible_addresses_first_internal_column,
        permissible_addresses_first_internal_column,
        permissible_function_gene_values,
        permissible_addresses_first_internal_column,
        permissible_addresses_first_internal_column,
        permissible_function_gene_values,
        permissible_addresses_second_internal_column,
        permissible_addresses_second_internal_column,
        permissible_function_gene_values,
        permissible_addresses_second_internal_column,
        permissible_addresses_second_internal_column,
        permissible_function_gene_values,
        permissible_addresses_second_internal_column,
        permissible_addresses_second_internal_column,
        permissible_function_gene_values,
        permissible_addresses_third_internal_column,
        permissible_addresses_third_internal_column,
        permissible_function_gene_values,
        permissible_addresses_third_internal_column,
        permissible_addresses_third_internal_column,
        permissible_function_gene_values,
        permissible_addresses_third_internal_column,
        permissible_addresses_third_internal_column,
        np.array(ID_OUTPUT_NODE),
        permissible_addresses_output,
        np.array(ID_NON_CODING_GENE),
    ]

    for (pv_per_gene_expected, pv_per_gene) in zip(
        permissible_values_expected, genome._permissible_values
    ):
        assert np.all(pv_per_gene_expected == pv_per_gene)


def test_mutate_does_not_reinitialize_parameters(genome_params, rng, mutation_rate):
    genome_params["primitives"] = (cgp.Parameter,)
    genome = cgp.Genome(**genome_params)
    genome.randomize(rng)
    genome._parameter_names_to_values["<p2>"] = math.pi
    parameter_names_to_values_before = genome._parameter_names_to_values.copy()
    genome.mutate(mutation_rate, rng)
    assert genome._parameter_names_to_values["<p2>"] == pytest.approx(
        parameter_names_to_values_before["<p2>"]
    )


def test_genome_reordering_empirically(rng):
    # empirically test that reordering does not change the output function of a genome

    pytest.importorskip("sympy")

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 14,
        "n_rows": 1,
        "primitives": (cgp.Mul, cgp.Sub, cgp.Add, cgp.ConstantFloat, cgp.Parameter),
    }

    genome = cgp.Genome(**genome_params)

    # f(x_0, x_1) = x_0 ** 2 - x_1 + 1 + 0.5
    dna_fixed = [
        ID_INPUT_NODE,  # x_0 (address 0)
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        ID_INPUT_NODE,  # x_1 (address 1)
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        0,  # Mul ->  x_0^2 (address 2)
        0,  # x
        0,  # x
        1,  # Sub -> x_0^2 - x_1 (address 3)
        2,  # x^2
        1,  # y
        1,  # Sub ->  0 (address 4)
        0,  # x
        0,  # x
        3,  # const -> 1 (address 5)
        2,
        3,
        3,  # const -> 1 (address 6)
        0,
        0,
        4,  # param -> 0.9 (address 7)
        4,
        3,
        4,  # param -> 0.4 (address 8)
        7,
        2,
        2,  # Add -> x_0^2 - x_1 + 1 (address 9)
        3,  # x_0^2 - x_1
        5,  # 1
        1,  # Sub -> x_0^2 - x_1 + 1 - 0.9 (address 10)
        9,  # x_0^2 - x_1 + 1
        7,  # 0.9
        2,  # Add -> x_0^2 - x_1 + 1 - 0.9 + 0.4 (address 11)
        10,  # x_0^2 - x_1 + 1 - 0.9
        8,  # 0.4
        3,  # const (address 12)
        0,
        1,
        3,  # const (address 13)
        0,
        1,
        3,  # const (address 14)
        0,
        1,
        3,  # const (address 15)
        0,
        1,
        ID_OUTPUT_NODE,
        11,
        ID_NON_CODING_GENE,
    ]

    genome.dna = dna_fixed
    genome._parameter_names_to_values["<p7>"] = 0.9
    genome._parameter_names_to_values["<p8>"] = 0.4

    sympy_expression = cgp.CartesianGraph(genome).to_sympy()
    n_reorderings = 100
    for _ in range(n_reorderings):
        genome.reorder(rng)
        new_graph = cgp.CartesianGraph(genome)
        sympy_expression_after_reorder = new_graph.to_sympy()
        assert sympy_expression_after_reorder == sympy_expression


def test_genome_reordering_parameterization_consistency(rng):

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 10,
        "n_rows": 2,
        "primitives": (cgp.Mul, cgp.Sub, cgp.Add, cgp.ConstantFloat),
    }

    genome = cgp.Genome(**genome_params)

    with pytest.raises(ValueError):
        genome.reorder(rng)

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 10,
        "n_rows": 1,
        "primitives": (cgp.Mul, cgp.Sub, cgp.Add, cgp.ConstantFloat),
        "levels_back": 5,
    }

    genome = cgp.Genome(**genome_params)

    with pytest.raises(ValueError):
        genome.reorder(rng)


def test_parameters_to_numpy_array():

    primitives = (cgp.Parameter,)
    genome = cgp.Genome(1, 1, 2, 1, primitives)
    # [f0(x), f1(x)] = [c1, c2]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        0,
        0,
        0,
        0,
        ID_OUTPUT_NODE,
        1,
    ]

    genome._parameter_names_to_values["<p1>"] = 0.8
    genome._parameter_names_to_values["<p2>"] = 0.9

    # all parameters
    expected_params = np.array([0.8, 0.9])
    expected_params_names = ["<p1>", "<p2>"]
    params, params_names = genome.parameters_to_numpy_array()
    assert np.all(params == pytest.approx(expected_params))
    assert params_names == expected_params_names

    # only parameters of active nodes
    expected_params = np.array([0.8])
    expected_params_names = ["<p1>"]
    params, params_names = genome.parameters_to_numpy_array(only_active_nodes=True)
    assert np.all(params == pytest.approx(expected_params))


def test_update_parameters_from_numpy_array():

    primitives = (cgp.Parameter,)
    genome = cgp.Genome(1, 1, 2, 1, primitives)
    # [f0(x), f1(x)] = [c1, c2]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        0,
        0,
        0,
        0,
        ID_OUTPUT_NODE,
        1,
    ]

    params = np.array([0.8, 0.9])
    params_names = ["<p1>", "<p2>"]
    any_parameter_changed = genome.update_parameters_from_numpy_array(params, params_names)

    assert any_parameter_changed
    assert genome._parameter_names_to_values["<p1>"] == pytest.approx(0.8)
    assert genome._parameter_names_to_values["<p2>"] == pytest.approx(0.9)

    # parameters do not change value
    any_parameter_changed = genome.update_parameters_from_numpy_array(params, params_names)

    assert not any_parameter_changed
    assert genome._parameter_names_to_values["<p1>"] == pytest.approx(0.8)
    assert genome._parameter_names_to_values["<p2>"] == pytest.approx(0.9)


def test_parameters_numpy_array_consistency():
    primitives = (cgp.Parameter,)
    genome = cgp.Genome(1, 1, 2, 1, primitives)
    # [f0(x), f1(x)] = [c1, c2]
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        0,
        0,
        0,
        0,
        ID_OUTPUT_NODE,
        1,
    ]

    genome._parameter_names_to_values["<p1>"] = 0.8
    genome._parameter_names_to_values["<p2>"] = 0.9

    genome.update_parameters_from_numpy_array(*genome.parameters_to_numpy_array())

    assert genome._parameter_names_to_values["<p1>"] == pytest.approx(0.8)
    assert genome._parameter_names_to_values["<p2>"] == pytest.approx(0.9)

    genome._parameter_names_to_values["<p1>"] = 1.1
    genome._parameter_names_to_values["<p2>"] = 1.2

    genome.update_parameters_from_numpy_array(
        *genome.parameters_to_numpy_array(only_active_nodes=True)
    )

    assert genome._parameter_names_to_values["<p1>"] == pytest.approx(1.1)
    assert genome._parameter_names_to_values["<p2>"] == pytest.approx(1.2)


def test_ncolumns_zero(rng):

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 0,
        "n_rows": 1,
        "primitives": (cgp.Mul, cgp.Sub, cgp.Add, cgp.ConstantFloat),
    }
    genome = cgp.Genome(**genome_params)
    genome.randomize(rng)

    CartesianGraph(genome).to_func()
    CartesianGraph(genome).to_numpy()
