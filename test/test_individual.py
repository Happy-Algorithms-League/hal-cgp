import math
import pickle
from collections import namedtuple

import numpy as np
import pytest

import cgp
from cgp import IndividualMultiGenome, IndividualSingleGenome
from cgp.genome import ID_INPUT_NODE, ID_NON_CODING_GENE, ID_OUTPUT_NODE

TestParams = namedtuple("TestParams", ["genome_params", "primitives", "dna", "target_function"])
params_list = [
    TestParams(
        genome_params={
            "n_inputs": 1,
            "n_outputs": 1,
            "n_columns": 2,
            "n_rows": 1,
            "levels_back": 2,
        },
        primitives=(cgp.Add, cgp.Parameter),
        dna=[
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
        ],
        target_function=lambda x, c: x + c,
    )
]
GraphInputValues = namedtuple("GraphInputValues", ["x", "c"])

graph_input_values_list = [GraphInputValues(x=[3.0, 5.0], c=1.0), GraphInputValues(x=[3.0], c=2.0)]


def _create_genome(genome_params, primitives, dna):
    genome = cgp.Genome(**genome_params, primitives=primitives)
    genome.dna = dna
    return genome


def _create_individual(genome, fitness=None, individual_type="SingleGenome"):
    if individual_type == "SingleGenome":
        ind = IndividualSingleGenome(genome)
        if fitness is not None:
            ind.fitness = fitness
        return ind
    elif individual_type == "MultiGenome":
        ind = IndividualMultiGenome([genome])
        if fitness is not None:
            ind.fitness = fitness
        return ind
    else:
        raise NotImplementedError("Unknown individual type.")


def _unpack_evaluation(value, individual_type="SingleGenome"):
    if individual_type == "SingleGenome":
        return value
    elif individual_type == "MultiGenome":
        return value[0]
    else:
        raise NotImplementedError("Unknown individual type.")


def _unpack_genome(individual, individual_type="SingleGenome"):
    if individual_type == "SingleGenome":
        return individual.genome
    elif individual_type == "MultiGenome":
        return individual.genome[0]
    else:
        raise NotImplementedError("Unknown individual type.")


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
def test_pickle_individual(individual_type):

    primitives = (cgp.Add,)
    genome = cgp.Genome(1, 1, 1, 1, primitives)
    individual = _create_individual(genome, individual_type=individual_type)

    with open("individual.pkl", "wb") as f:
        pickle.dump(individual, f)


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
@pytest.mark.parametrize("params", params_list)
@pytest.mark.parametrize("graph_input_values", graph_input_values_list)
def test_individual_with_parameter_python(individual_type, params, graph_input_values):
    genome_params, primitives, dna, target_function = params
    genome = _create_genome(genome_params, primitives, dna)
    individual = _create_individual(genome, individual_type=individual_type)

    x, c = graph_input_values.x, graph_input_values.c
    _unpack_genome(individual, individual_type)._parameter_names_to_values["<p1>"] = c
    f = _unpack_evaluation(individual.to_func(), individual_type)

    for xi in x:
        y = f(xi)
        assert y == pytest.approx(target_function(xi, c))


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
@pytest.mark.parametrize("params", params_list)
@pytest.mark.parametrize("graph_input_values", graph_input_values_list)
def test_individual_with_parameter_torch(individual_type, params, graph_input_values):
    torch = pytest.importorskip("torch")
    genome_params, primitives, dna, target_function = params
    genome = _create_genome(genome_params, primitives, dna)
    individual = _create_individual(genome, individual_type=individual_type)

    x, c = torch.tensor(graph_input_values.x).unsqueeze(1), graph_input_values.c
    _unpack_genome(individual, individual_type)._parameter_names_to_values["<p1>"] = c
    f = _unpack_evaluation(individual.to_torch(), individual_type)
    y = f(x)

    for i in range(x.shape[0]):
        assert y[i, 0].item() == pytest.approx(target_function(x[i, 0].item(), c))


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
@pytest.mark.parametrize("params", params_list)
@pytest.mark.parametrize("graph_input_values", graph_input_values_list)
def test_individual_with_parameter_sympy(individual_type, params, graph_input_values):
    pytest.importorskip("sympy")
    genome_params, primitives, dna, target_function = params
    genome = _create_genome(genome_params, primitives, dna)
    individual = _create_individual(genome, individual_type=individual_type)

    x, c = graph_input_values.x, graph_input_values.c
    _unpack_genome(individual, individual_type)._parameter_names_to_values["<p1>"] = c
    f = _unpack_evaluation(individual.to_sympy(), individual_type)

    for xi in x:
        y = f.subs("x_0", xi).evalf()
        assert y == pytest.approx(target_function(xi, c))


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
@pytest.mark.parametrize("params", params_list)
@pytest.mark.parametrize("graph_input_values", graph_input_values_list)
def test_individual_with_parameter_numpy(individual_type, params, graph_input_values):
    genome_params, primitives, dna, target_function = params
    genome = _create_genome(genome_params, primitives, dna)
    individual = _create_individual(genome, individual_type=individual_type)

    x, c = np.array(graph_input_values.x)[:, np.newaxis], graph_input_values.c

    _unpack_genome(individual, individual_type)._parameter_names_to_values["<p1>"] = c
    f = _unpack_evaluation(individual.to_numpy(), individual_type)

    y = f(x)
    for i in range(x.shape[0]):
        assert y[i, 0].item() == pytest.approx(target_function(x[i, 0].item(), c))


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
def test_to_and_from_torch_plus_backprop(individual_type):
    torch = pytest.importorskip("torch")
    primitives = (cgp.Mul, cgp.Parameter)
    genome = cgp.Genome(1, 1, 2, 2, primitives, 1)
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
    individual = _create_individual(genome, individual_type=individual_type)

    def f_target(x):
        return math.pi * x

    f = individual.to_torch()
    f_opt = f if individual_type == "SingleGenome" else f[0]
    optimizer = torch.optim.SGD(f_opt.parameters(), lr=1e-1)

    criterion = torch.nn.MSELoss()

    for i in range(200):

        x = torch.DoubleTensor(1, 1).normal_()
        y = f_opt(x)

        y_target = f_target(x)

        loss = criterion(y, y_target)
        f_opt.zero_grad()
        loss.backward()

        optimizer.step()

    assert loss.detach().numpy() < 1e-15

    # use old parameter values to compile function
    x = 3.0
    f_func = _unpack_evaluation(individual.to_func(), individual_type)
    y = f_func(x)
    assert y != pytest.approx(f_target(x))

    # update parameter values from torch class and compile new
    # function with new parameter values
    individual.update_parameters_from_torch_class(f)
    f_func = _unpack_evaluation(individual.to_func(), individual_type)
    y = f_func(x)
    assert y == pytest.approx(f_target(x))


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
def test_update_parameters_from_torch_class_resets_fitness(individual_type):
    pytest.importorskip("torch")
    primitives = (cgp.Mul, cgp.Parameter)
    genome = cgp.Genome(1, 1, 2, 1, primitives, 1)
    # f(x) = c * x
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
    fitness = 1.0
    individual = _create_individual(genome, fitness=fitness, individual_type=individual_type)

    f = individual.to_torch()
    f_opt = _unpack_evaluation(f, individual_type=individual_type)
    f_opt._p1.data[0] = math.pi

    assert not individual.fitness_is_None()
    individual.update_parameters_from_torch_class(f)
    assert individual.fitness_is_None()

    g = _unpack_evaluation(individual.to_func(), individual_type)
    x = 2.0
    assert g(x) == pytest.approx(math.pi * x)


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
def test_update_parameters_from_torch_class_does_not_reset_fitness_for_unused_parameters(
    individual_type,
):
    pytest.importorskip("torch")
    primitives = (cgp.Mul, cgp.Parameter)
    genome = cgp.Genome(1, 1, 2, 1, primitives, 1)
    # f(x) = x ** 2
    genome.dna = [
        ID_INPUT_NODE,
        ID_NON_CODING_GENE,
        ID_NON_CODING_GENE,
        1,  # these
        0,  # three
        0,  # genes code for an unused parameter node
        0,
        0,
        0,
        ID_OUTPUT_NODE,
        2,
        ID_NON_CODING_GENE,
    ]
    fitness = 1.0
    individual = _create_individual(genome, fitness=fitness, individual_type=individual_type)

    f = individual.to_torch()

    assert not individual.fitness_is_None()
    individual.update_parameters_from_torch_class(f)
    assert not individual.fitness_is_None()

    g = _unpack_evaluation(individual.to_func(), individual_type)
    x = 2.0
    assert g(x) == pytest.approx(x ** 2)


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
def test_individual_randomize_genome(individual_type, rng_seed):
    rng = np.random.RandomState(rng_seed)
    primitives = (cgp.Add, cgp.Mul)
    genome = cgp.Genome(1, 1, 2, 1, primitives, 1)
    genome.randomize(rng)

    dna_old = list(genome.dna)
    individual = _create_individual(genome, individual_type=individual_type)

    individual.randomize_genome(rng)
    assert dna_old != _unpack_genome(individual, individual_type)


@pytest.mark.parametrize("individual_type", ["SingleGenome", "MultiGenome"])
def test_clone_copies_user_defined_attributes(individual_type, genome_params, rng):
    genome = cgp.Genome(**genome_params)
    genome.randomize(rng)
    ind = _create_individual(genome, individual_type=individual_type)
    my_attribute = "this is a custom attribute"
    ind.my_attribute = my_attribute
    ind_clone = ind.clone()

    assert ind.my_attribute == my_attribute
    assert ind_clone.my_attribute == my_attribute
