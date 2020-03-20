import math
import pickle
import pytest
import torch

import gp
from gp.individual import Individual
from gp.genome import ID_INPUT_NODE, ID_OUTPUT_NODE, ID_NON_CODING_GENE


def test_pickle_individual():

    primitives = [gp.Add]
    genome = gp.Genome(1, 1, 1, 1, 1, primitives)
    individual = Individual(None, genome)

    with open("individual.pkl", "wb") as f:
        pickle.dump(individual, f)


def test_individual_with_parameter_python():

    primitives = [gp.Add, gp.Parameter]
    genome = gp.Genome(1, 1, 2, 1, 2, primitives)
    # f(x) = x + c
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        1,
        0,
        0,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    individual = Individual(None, genome)

    c = 1.0
    x = [3.0]

    f = individual.to_func()
    y = f(x)

    assert y[0] == pytest.approx(x[0] + c)

    c = 2.0
    individual.parameter_names_to_values["<p1>"] = c

    f = individual.to_func()
    y = f(x)

    assert y[0] == pytest.approx(x[0] + c)


def test_individual_with_parameter_torch():

    primitives = [gp.Add, gp.Parameter]
    genome = gp.Genome(1, 1, 2, 1, 2, primitives)
    # f(x) = x + c
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        1,
        0,
        0,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    individual = Individual(None, genome)

    c = 1.0
    x = torch.empty(2, 1).normal_()

    f = individual.to_torch()
    y = f(x)

    assert y[0, 0].item() == pytest.approx(x[0, 0].item() + c)
    assert y[1, 0].item() == pytest.approx(x[1, 0].item() + c)

    c = 2.0
    individual.parameter_names_to_values["<p1>"] = c

    f = individual.to_torch()
    y = f(x)

    assert y[0, 0].item() == pytest.approx(x[0, 0].item() + c)
    assert y[1, 0].item() == pytest.approx(x[1, 0].item() + c)


def test_individual_with_parameter_sympy():

    primitives = [gp.Add, gp.Parameter]
    genome = gp.Genome(1, 1, 2, 1, 2, primitives)
    # f(x) = x + c
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        1,
        0,
        0,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    individual = Individual(None, genome)

    c = 1.0
    x = [3.0]

    f = individual.to_sympy()[0]
    y = f.subs("x_0", x[0]).evalf()

    assert y == pytest.approx(x[0] + c)

    c = 2.0
    individual.parameter_names_to_values["<p1>"] = c

    f = individual.to_sympy()[0]
    y = f.subs("x_0", x[0]).evalf()

    assert y == pytest.approx(x[0] + c)


def test_to_and_from_torch_plus_backprop():
    primitives = [gp.Mul, gp.Parameter]
    genome = gp.Genome(1, 1, 2, 2, 1, primitives)
    # f(x) = c * x
    genome.dna = [
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
        1,
        0,
        0,
        1,
        ID_OUTPUT_NODE,
        3,
        ID_NON_CODING_GENE,
    ]
    individual = Individual(None, genome)

    def f_target(x):
        return math.pi * x

    f = individual.to_torch()

    optimizer = torch.optim.SGD(f.parameters(), lr=1e-1)
    criterion = torch.nn.MSELoss()

    for i in range(200):

        x = torch.DoubleTensor(1, 1).normal_()
        y = f(x)

        y_target = f_target(x)

        loss = criterion(y, y_target)
        f.zero_grad()
        loss.backward()

        optimizer.step()

    assert loss.detach().numpy() < 1e-15

    # use old parameter values to compile function
    x = [3.0]
    f_func = individual.to_func()
    y = f_func(x)
    assert y[0] != pytest.approx(f_target(x[0]))

    # update parameter values from torch class and compile new
    # function with new parameter values
    individual.update_parameters_from_torch_class(f)
    f_func = individual.to_func()
    y = f_func(x)
    assert y[0] == pytest.approx(f_target(x[0]))
