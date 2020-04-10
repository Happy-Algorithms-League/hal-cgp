import pytest

import gp


def test_constant_float():
    params = {"n_inputs": 2, "n_outputs": 1, "n_columns": 1, "n_rows": 1, "levels_back": 1}

    val = 1.678

    primitives = [gp.node_factories.ConstantFloatFactory(val)]
    genome = gp.Genome(
        params["n_inputs"],
        params["n_outputs"],
        params["n_columns"],
        params["n_rows"],
        params["levels_back"],
        primitives,
    )
    genome.dna = [-1, None, -1, None, 0, None, -2, 2]
    graph = gp.CartesianGraph(genome)

    x = [None, None]
    y = graph(x)

    assert pytest.approx(val == y[0])

    # make sure different classes are created for multiple calls to the class
    # factory
    prim_0 = gp.node_factories.ConstantFloatFactory(val)(0, [None])
    prim_1 = gp.node_factories.ConstantFloatFactory(val)(0, [None])

    assert prim_0 is not prim_1

    prim_1._output = 2 * val

    assert pytest.approx(val == prim_0._output)
    assert pytest.approx(2 * val == prim_1._output)
