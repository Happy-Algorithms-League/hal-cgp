import math
import numpy as np
import pickle
import pytest
import torch

import gp
from gp.cgp_individual import CGPIndividual


SEED = np.random.randint(2 ** 31)


def test_pickle_individual():

    primitives = [gp.CGPAdd]
    genome = gp.CGPGenome(1, 1, 1, 1, 1, primitives)
    individual = CGPIndividual(None, genome)

    with open("individual.pkl", "wb") as f:
        pickle.dump(individual, f)


def test_individual_with_parameter_python():

    primitives = [gp.CGPAdd, gp.CGPParameter]
    genome = gp.CGPGenome(1, 1, 2, 1, 2, primitives)
    # output uses the CGPParameter node, y = x + c; c is initialized
    # to zero
    genome.dna = [-1, None, None, 1, None, None, 0, 0, 1, -2, 2, None]
    individual = CGPIndividual(None, genome)

    x = [3.0]

    f = individual.to_func()
    y = f(x)

    assert y[0] == pytest.approx(x[0])

    c = 1.0
    individual.parameter_names_to_values["<p1>"] = c

    f = individual.to_func()
    y = f(x)

    assert y[0] == pytest.approx(x[0] + c)


def test_individual_with_parameter_pytorch():

    primitives = [gp.CGPAdd, gp.CGPParameter]
    genome = gp.CGPGenome(1, 1, 2, 1, 2, primitives)
    # output uses the CGPParameter node, y = x + c; c is initialized
    # to zero
    genome.dna = [-1, None, None, 1, None, None, 0, 0, 1, -2, 2, None]
    individual = CGPIndividual(None, genome)

    x = torch.empty(2, 1).normal_()

    f = individual.to_torch()
    y = f(x)

    assert y[0, 0].item() == pytest.approx(x[0, 0].item())
    assert y[1, 0].item() == pytest.approx(x[1, 0].item())

    c = 1.0
    individual.parameter_names_to_values["<p1>"] = c

    f = individual.to_torch()
    y = f(x)

    assert y[0, 0].item() == pytest.approx(x[0, 0].item() + c)
    assert y[1, 0].item() == pytest.approx(x[1, 0].item() + c)


def test_individual_with_parameter_sympy():

    primitives = [gp.CGPAdd, gp.CGPParameter]
    genome = gp.CGPGenome(1, 1, 2, 1, 2, primitives)
    # output uses the CGPParameter node, y = x + c; c is initialized
    # to zero
    genome.dna = [-1, None, None, 1, None, None, 0, 0, 1, -2, 2, None]
    individual = CGPIndividual(None, genome)

    x = [3.0]

    f = individual.to_sympy()[0]
    y = f.subs("x_0", x[0]).evalf()

    assert y == pytest.approx(x[0])

    c = 1.0
    individual.parameter_names_to_values["<p1>"] = c

    f = individual.to_sympy()[0]
    y = f.subs("x_0", x[0]).evalf()

    assert y == pytest.approx(x[0] + c)


def test_to_and_from_torch_plus_backprop():
    primitives = [gp.CGPMul, gp.CGPParameter]
    genome = gp.CGPGenome(1, 1, 2, 2, 1, primitives)
    genome.dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 3, None]
    individual = CGPIndividual(None, genome)

    c = individual.to_torch()

    optimizer = torch.optim.SGD(c.parameters(), lr=1e-1)
    criterion = torch.nn.MSELoss()

    for i in range(200):

        x = torch.Tensor(1, 1).normal_()
        y = c(x)

        y_target = math.pi * x

        loss = criterion(y, y_target)
        c.zero_grad()
        loss.backward()

        optimizer.step()

    assert loss.detach().numpy() < 1e-15

    # use old parameter values to compile function
    x = [3.0]
    f = individual.to_func()
    y = f(x)
    assert y[0] != pytest.approx(math.pi * x[0])

    # update parameter values from torch class and compile new
    # function with new parameter values
    individual.update_parameters_from_torch_class(c)
    f = individual.to_func()
    y = f(x)
    assert y[0] == pytest.approx(math.pi * x[0])
