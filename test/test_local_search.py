import pytest

import cgp
from cgp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE


def test_gradient_based_step_towards_maximum():
    torch = pytest.importorskip("torch")

    primitives = (cgp.Parameter,)
    genome = cgp.Genome(1, 1, 1, 1, 1, primitives)
    # f(x) = c
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]
    ind = cgp.individual.IndividualSingleGenome(None, genome)

    def objective(f):
        x_dummy = torch.zeros((1, 1), dtype=torch.double)  # not used
        target_value = torch.ones((1, 1), dtype=torch.double)
        return torch.nn.MSELoss()(f(x_dummy), target_value)

    # test increase parameter value if too small
    ind.genome.parameter_names_to_values["<p1>"] = 0.9
    cgp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.genome.parameter_names_to_values["<p1>"] == pytest.approx(0.91)

    # test decrease parameter value if too large
    ind.genome.parameter_names_to_values["<p1>"] = 1.1
    cgp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.genome.parameter_names_to_values["<p1>"] == pytest.approx(1.09)

    # test no change of parameter value if at optimum
    ind.genome.parameter_names_to_values["<p1>"] = 1.0
    cgp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.genome.parameter_names_to_values["<p1>"] == pytest.approx(1.0)


def test_gradient_based_step_towards_maximum_multi_genome():
    torch = pytest.importorskip("torch")

    primitives = (cgp.Parameter,)
    genome = cgp.Genome(1, 1, 1, 1, 1, primitives)
    # f(x) = c
    genome.dna = [ID_INPUT_NODE, ID_NON_CODING_GENE, 0, 0, ID_OUTPUT_NODE, 1]
    genome2 = genome.clone()
    ind = cgp.individual.IndividualMultiGenome(None, [genome, genome2])

    def objective(f):
        x_dummy = torch.zeros((1, 1), dtype=torch.double)  # not used
        target_value = torch.ones((1, 1), dtype=torch.double)
        loss = torch.nn.MSELoss()(f[0](x_dummy), target_value) + 2 * torch.nn.MSELoss()(
            f[1](x_dummy), target_value
        )
        return loss

    # test increase parameter value if too small
    ind.genome[0].parameter_names_to_values["<p1>"] = 0.9
    ind.genome[1].parameter_names_to_values["<p1>"] = 0.9
    cgp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.genome[0].parameter_names_to_values["<p1>"] == pytest.approx(0.91)
    assert ind.genome[1].parameter_names_to_values["<p1>"] == pytest.approx(0.92)

    # test decrease parameter value if too large
    ind.genome[0].parameter_names_to_values["<p1>"] = 1.1
    ind.genome[1].parameter_names_to_values["<p1>"] = 1.1
    cgp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.genome[0].parameter_names_to_values["<p1>"] == pytest.approx(1.09)
    assert ind.genome[1].parameter_names_to_values["<p1>"] == pytest.approx(1.08)

    # test no change of parameter value if at optimum
    ind.genome[0].parameter_names_to_values["<p1>"] = 1.0
    ind.genome[1].parameter_names_to_values["<p1>"] = 1.0
    cgp.local_search.gradient_based(ind, objective, 0.05, 1)
    assert ind.genome[0].parameter_names_to_values["<p1>"] == pytest.approx(1.0)
    assert ind.genome[1].parameter_names_to_values["<p1>"] == pytest.approx(1.0)
