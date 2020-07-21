import numpy as np
import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE
from cgp.cartesian_graph import CartesianGraph


def test_check_dna_consistency():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
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

    # invalid input gene for input node
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

    # invalid input gene for hidden node
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

    # invalid input gene for input node
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

    # invalid inactive input gene for output node
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


def test_permissible_inputs():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 4, "n_rows": 3, "levels_back": 2}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.randomize(np.random)

    for input_idx in range(params["n_inputs"]):
        region_idx = input_idx
        with pytest.raises(AssertionError):
            genome._permissible_inputs(region_idx)

    expected_for_hidden = [
        [0, 1],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 5, 6, 7, 8, 9, 10],
    ]

    for column_idx in range(params["n_columns"]):
        region_idx = params["n_inputs"] + params["n_rows"] * column_idx
        assert expected_for_hidden[column_idx] == genome._permissible_inputs(region_idx)

    expected_for_output = list(range(params["n_inputs"] + params["n_rows"] * params["n_columns"]))

    for output_idx in range(params["n_outputs"]):
        region_idx = params["n_inputs"] + params["n_rows"] * params["n_columns"] + output_idx
        assert expected_for_output == genome._permissible_inputs(region_idx)


def test_region_iterators():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = (cgp.Add,)
    genome = cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
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
            params["levels_back"],
            primitives,
        )

    params["levels_back"] = params["n_columns"] + 1
    with pytest.raises(ValueError):
        cgp.Genome(
            params["n_inputs"],
            params["n_outputs"],
            params["n_columns"],
            params["n_rows"],
            params["levels_back"],
            primitives,
        )

    params["levels_back"] = params["n_columns"] - 1
    cgp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )


def test_catch_invalid_allele_in_inactive_region():
    primitives = (cgp.ConstantFloat,)
    genome = cgp.Genome(1, 1, 1, 1, 1, primitives)

    # should raise error: ConstantFloat node has no inputs, but silent
    # input gene should still specify valid input
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
    genome = cgp.Genome(2, 1, 2, 1, None, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_input_region(0)
    assert not genome._is_gene_in_input_region(6)


def test_is_gene_in_hidden_region(rng_seed):
    genome = cgp.Genome(2, 1, 2, 1, None, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_hidden_region(6)
    assert genome._is_gene_in_hidden_region(9)
    assert not genome._is_gene_in_hidden_region(5)
    assert not genome._is_gene_in_hidden_region(12)


def test_is_gene_in_output_region(rng_seed):
    genome = cgp.Genome(2, 1, 2, 1, None, (cgp.Add,))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_output_region(12)
    assert not genome._is_gene_in_output_region(11)


def test_mutation_rate(rng_seed, mutation_rate):
    n_inputs = 1
    n_outputs = 1
    n_columns = 4
    n_rows = 3
    genome = cgp.Genome(n_inputs, n_outputs, n_columns, n_rows, None, (cgp.Add, cgp.Sub))
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    def count_n_immutable_genes(n_inputs, n_outputs, n_rows):
        length_per_region = genome.primitives.max_arity + 1  # function gene + input gene addresses
        n_immutable_genes = n_inputs * length_per_region  # none of the input genes are mutable
        n_immutable_genes += n_outputs * (
            length_per_region - 1
        )  # only one gene per output can be mutated
        if n_inputs == 1:
            n_immutable_genes += (
                n_rows * genome.primitives.max_arity
            )  # input gene addresses in the first (hidden) column can't be mutated
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
    length_per_region = genome.primitives.max_arity + 1  # function gene + input gene addresses
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


def test_permissible_values_hidden(rng_seed):
    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": (cgp.Add, cgp.Sub, cgp.ConstantFloat),
    }

    genome = cgp.Genome(**genome_params)

    # test function gene
    gene_idx = 6  # function gene of first hidden region
    allele = 0  #
    region_idx = None  # can be any number, since not touched for function gene
    permissible_func_gene_values = genome._determine_alternative_permissible_values_hidden_gene(
        gene_idx, allele, region_idx
    )
    assert permissible_func_gene_values == [
        1,
        2,
    ]  # function idx 1,2 (cgp.Sub, cgp.ConstantFloat)

    # test input gene
    gene_idx = 16  # first input gene in second column of hidden region
    allele = 0
    region_idx = gene_idx // (
        genome.primitives.max_arity + 1
    )  # function gene + input gene addresses
    permissible_input_gene_values = genome._determine_alternative_permissible_values_hidden_gene(
        gene_idx, allele, region_idx
    )

    assert permissible_input_gene_values == [
        1,
        2,
        3,
        4,
    ]  # gene indices of input, 1st hidden layer


def test_permissible_values_output(rng_seed):
    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 3,
        "n_rows": 3,
        "levels_back": 2,
        "primitives": (cgp.Add, cgp.Sub, cgp.ConstantFloat),
    }
    genome = cgp.Genome(**genome_params)

    gene_idx_function = 33
    gene_idx_input0 = 34
    gene_idx_input1 = 35

    allele = 5  # set current value for input0 -> should be excluded from permissible values

    permissible_values = genome._determine_alternative_permissible_values_output_gene(
        gene_idx_function, allele
    )
    assert permissible_values == []

    permissible_values = genome._determine_alternative_permissible_values_output_gene(
        gene_idx_input0, allele
    )
    assert permissible_values == [0, 1, 2, 3, 4, 6, 7, 8, 9, 10]

    permissible_values = genome._determine_alternative_permissible_values_output_gene(
        gene_idx_input1, allele
    )
    assert permissible_values == []
