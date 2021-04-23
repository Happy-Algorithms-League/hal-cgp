import numpy as np
import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_NON_CODING_GENE, ID_OUTPUT_NODE


def test_step_towards_maximum(rng_seed):
    primitives = (cgp.Parameter,)
    genome = cgp.Genome(1, 2, 2, 1, primitives)
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
        ID_OUTPUT_NODE,
        2,
    ]
    ind = cgp.individual.IndividualSingleGenome(genome)
    ind.idx = 0

    def objective(ind):
        f = ind.to_numpy()
        x_dummy = np.zeros((1, 1))  # input, not used
        target_value = np.array([[1.0, 1.1]])
        return -np.sum((f(x_dummy) - target_value) ** 2)

    # test increase parameter value if too small
    ind.genome._parameter_names_to_values["<p1>"] = 0.8
    ind.genome._parameter_names_to_values["<p2>"] = 0.9
    es = cgp.local_search.EvolutionStrategies(objective, rng_seed, max_steps=1)
    es(ind)
    assert ind.genome._parameter_names_to_values["<p1>"] > 0.8
    assert ind.genome._parameter_names_to_values["<p2>"] > 0.9

    # test decrease parameter value if too large
    ind.genome._parameter_names_to_values["<p1>"] = 1.1
    ind.genome._parameter_names_to_values["<p2>"] = 1.2
    es = cgp.local_search.EvolutionStrategies(objective, rng_seed, max_steps=1)
    es(ind)
    assert ind.genome._parameter_names_to_values["<p1>"] < 1.1
    assert ind.genome._parameter_names_to_values["<p2>"] < 1.2


def _objective_convergence_to_maximum(ind):
    f = ind.to_numpy()
    x_dummy = np.zeros(1)  # input, not used
    target_value_0 = 1.0
    target_value_1 = 1.1
    y = f(x_dummy)
    return -((y[0] - target_value_0) ** 2) - (y[1] - target_value_1) ** 2


def test_convergence_to_maximum(rng_seed):
    primitives = (cgp.Parameter,)
    genome = cgp.Genome(1, 2, 2, 1, primitives)
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
        ID_OUTPUT_NODE,
        2,
    ]
    ind = cgp.individual.IndividualSingleGenome(genome)
    ind.idx = 0

    # single process
    ind.genome._parameter_names_to_values["<p1>"] = 0.85
    ind.genome._parameter_names_to_values["<p2>"] = 0.95
    es = cgp.local_search.EvolutionStrategies(
        _objective_convergence_to_maximum, rng_seed, max_steps=60
    )
    es(ind)
    assert ind.genome._parameter_names_to_values["<p1>"] == pytest.approx(1.0)
    assert ind.genome._parameter_names_to_values["<p2>"] == pytest.approx(1.1)


def test_step_towards_maximum_multi_genome(rng_seed):
    primitives = (cgp.Parameter,)
    genome = cgp.Genome(1, 2, 2, 1, primitives)
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
        ID_OUTPUT_NODE,
        2,
    ]
    genome2 = genome.clone()
    ind = cgp.individual.IndividualMultiGenome([genome, genome2])
    ind.idx = 0

    def objective(ind):
        f = ind.to_numpy()
        x_dummy = np.zeros(1)  # input, not used
        target_value_0 = 1.0
        target_value_1 = 1.1
        y0 = f[0](x_dummy)
        y1 = f[1](x_dummy)
        return (
            -((y0[0] - target_value_0) ** 2)
            - (y0[1] - target_value_1) ** 2
            - (y1[0] - target_value_0) ** 2
            - (y1[1] - target_value_1) ** 2
        )

    # test increase parameter value if too small first genome,
    # decrease parameter value if too large for second genome
    ind.genome[0]._parameter_names_to_values["<p1>"] = 0.8
    ind.genome[0]._parameter_names_to_values["<p2>"] = 0.9
    ind.genome[1]._parameter_names_to_values["<p1>"] = 1.1
    ind.genome[1]._parameter_names_to_values["<p2>"] = 1.2
    es = cgp.local_search.EvolutionStrategies(objective, rng_seed, max_steps=2)
    es(ind)
    assert ind.genome[0]._parameter_names_to_values["<p1>"] > 0.8
    assert ind.genome[0]._parameter_names_to_values["<p2>"] > 0.9
    assert ind.genome[1]._parameter_names_to_values["<p1>"] < 1.1
    assert ind.genome[1]._parameter_names_to_values["<p2>"] < 1.2

    # test decrease parameter value if too large first genome,
    # increase parameter value if too small for second genome
    ind.genome[0]._parameter_names_to_values["<p1>"] = 1.1
    ind.genome[0]._parameter_names_to_values["<p2>"] = 1.2
    ind.genome[1]._parameter_names_to_values["<p1>"] = 0.8
    ind.genome[1]._parameter_names_to_values["<p2>"] = 0.9
    es = cgp.local_search.EvolutionStrategies(objective, rng_seed, max_steps=2)
    es(ind)
    assert ind.genome[0]._parameter_names_to_values["<p1>"] < 1.1
    assert ind.genome[0]._parameter_names_to_values["<p2>"] < 1.2
    assert ind.genome[1]._parameter_names_to_values["<p1>"] > 0.8
    assert ind.genome[1]._parameter_names_to_values["<p2>"] > 0.9
