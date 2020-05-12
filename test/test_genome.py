import numpy as np
import pytest

import gp
from gp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE


def test_check_dna_consistency():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = [gp.Add]
    genome = gp.Genome(
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


def test_permissable_inputs():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 4, "n_rows": 3, "levels_back": 2}

    primitives = [gp.Add]
    genome = gp.Genome(
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
            genome._permissable_inputs(region_idx)

    expected_for_hidden = [
        [0, 1],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 5, 6, 7, 8, 9, 10],
    ]

    for column_idx in range(params["n_columns"]):
        region_idx = params["n_inputs"] + params["n_rows"] * column_idx
        assert expected_for_hidden[column_idx] == genome._permissable_inputs(region_idx)

    expected_for_output = list(range(params["n_inputs"] + params["n_rows"] * params["n_columns"]))

    for output_idx in range(params["n_outputs"]):
        region_idx = params["n_inputs"] + params["n_rows"] * params["n_columns"] + output_idx
        assert expected_for_output == genome._permissable_inputs(region_idx)


def test_region_iterators():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    primitives = [gp.Add]
    genome = gp.Genome(
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

    primitives = [gp.Add]

    params["levels_back"] = 0
    with pytest.raises(ValueError):
        gp.Genome(
            params["n_inputs"],
            params["n_outputs"],
            params["n_columns"],
            params["n_rows"],
            params["levels_back"],
            primitives,
        )

    params["levels_back"] = params["n_columns"] + 1
    with pytest.raises(ValueError):
        gp.Genome(
            params["n_inputs"],
            params["n_outputs"],
            params["n_columns"],
            params["n_rows"],
            params["levels_back"],
            primitives,
        )

    params["levels_back"] = params["n_columns"] - 1
    gp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )


def test_catch_invalid_allele_in_inactive_region():
    primitives = [gp.ConstantFloat]
    genome = gp.Genome(1, 1, 1, 1, 1, primitives)

    # should raise error: ConstantFloat node has no inputs, but silent
    # input gene should still specify valid input
    with pytest.raises(ValueError):
        genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, ID_NON_CODING_GENE, ID_OUTPUT_NODE, 1]

    # correct
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]


def test_individuals_have_different_genomes(rng_seed):

    population_params = {
        "n_parents": 5,
        "n_offspring": 5,
        "generations": 50000,
        "n_breeding": 5,
        "tournament_size": 2,
        "mutation_rate": 0.05,
    }

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 6,
        "n_rows": 6,
        "levels_back": 2,
        "primitives": [gp.Add, gp.Sub, gp.Mul, gp.Div, gp.ConstantFloat],
    }

    def objective(ind):
        ind.fitness = ind.idx
        return ind

    pop = gp.Population(
        population_params["n_parents"], population_params["mutation_rate"], rng_seed, genome_params
    )
    ea = gp.ea.MuPlusLambda(
        population_params["n_offspring"],
        population_params["n_breeding"],
        population_params["tournament_size"],
    )

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
    genome = gp.Genome(2, 1, 2, 1, None, [gp.Add])
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_input_region(0)
    assert not genome._is_gene_in_input_region(6)


def test_is_gene_in_hidden_region(rng_seed):
    genome = gp.Genome(2, 1, 2, 1, None, [gp.Add])
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_hidden_region(6)
    assert genome._is_gene_in_hidden_region(9)
    assert not genome._is_gene_in_hidden_region(5)
    assert not genome._is_gene_in_hidden_region(12)


def test_is_gene_in_output_region(rng_seed):
    genome = gp.Genome(2, 1, 2, 1, None, [gp.Add])
    rng = np.random.RandomState(rng_seed)
    genome.randomize(rng)

    assert genome._is_gene_in_output_region(12)
    assert not genome._is_gene_in_output_region(11)


def test_mutate_hidden_region(rng_seed):
    rng = np.random.RandomState(rng_seed)
    genome = gp.Genome(1, 1, 3, 1, None, [gp.Add, gp.ConstantFloat])
    dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
        2,
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
    ]
    genome.dna = list(dna)
    active_regions = gp.CartesianGraph(genome).determine_active_regions()

    # mutating any gene in inactive region returns True
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(3, active_regions, rng) is True
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(4, active_regions, rng) is True
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(5, active_regions, rng) is True

    # mutating function gene in active region returns False
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(6, active_regions, rng) is False
    # mutating inactive genes in active region returns True
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(7, active_regions, rng) is True
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(8, active_regions, rng) is True

    # mutating any gene in active region without silent genes returns False
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(9, active_regions, rng) is False
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(10, active_regions, rng) is False
    genome.dna = list(dna)
    assert genome._mutate_hidden_region(11, active_regions, rng) is False


def test_mutate_output_region(rng_seed):
    rng = np.random.RandomState(rng_seed)
    genome = gp.Genome(1, 1, 2, 1, None, [gp.Add])
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
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

    assert genome._mutate_output_region(9, rng) is False
    assert genome._mutate_output_region(10, rng) is True
    assert genome._mutate_output_region(11, rng) is False
