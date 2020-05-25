import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE


def test_constant_float():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    val = 1.678

    primitives = (cgp.node_factories.ConstantFloatFactory(val),)
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
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        0,
        0,
        ID_OUTPUT_NODE,
        2,
    ]
    graph = cgp.CartesianGraph(genome)

    x = [None, None]
    y = graph(x)

    assert val == pytest.approx(y[0])

    # make sure different classes are created for multiple calls to the class
    # factory
    prim_0 = cgp.node_factories.ConstantFloatFactory(val)(0, [None])
    prim_1 = cgp.node_factories.ConstantFloatFactory(val)(0, [None])

    assert prim_0 is not prim_1

    prim_1._output = 2 * val

    assert val == pytest.approx(prim_0._output)
    assert 2 * val == pytest.approx(prim_1._output)
