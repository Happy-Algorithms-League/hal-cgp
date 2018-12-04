import numpy as np
import matplotlib.pyplot as plt
import pickle
import pytest
import sympy
import sys
import torch

sys.path.insert(0, '../')
import gp


def test_check_levels_back_consistency():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 4,
        'n_rows': 3,
        'levels_back': None,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])

    params['levels_back'] = 0
    with pytest.raises(ValueError):
        gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)

    params['levels_back'] = params['n_columns'] + 1
    with pytest.raises(ValueError):
        gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)

    params['levels_back'] = params['n_columns'] - 1
    gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)


# -> genome
def test_permissable_inputs():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 4,
        'n_rows': 3,
        'levels_back': 2,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.randomize()

    for input_idx in range(params['n_inputs']):
        region_idx = input_idx
        with pytest.raises(AssertionError):
            genome._permissable_inputs(region_idx)

    expected_for_hidden = [
        [0, 1],
        [0, 1, 2, 3, 4],
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 1, 5, 6, 7, 8, 9, 10],
    ]

    for column_idx in range(params['n_columns']):
        region_idx = params['n_inputs'] + params['n_rows'] * column_idx
        assert expected_for_hidden[column_idx] == genome._permissable_inputs(region_idx)

    expected_for_output = list(range(params['n_inputs'] + params['n_rows'] * params['n_columns']))

    for output_idx in range(params['n_outputs']):
        region_idx = params['n_inputs'] + params['n_rows'] * params['n_columns'] + output_idx
        assert expected_for_output == genome._permissable_inputs(region_idx)


# -> genome
def test_region_generators():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 0, None]

    for region_idx, region in genome.input_regions():
        assert(region == [-1, None, None])

    for region_idx, region in genome.hidden_regions():
        assert(region == [0, 0, 1])

    for region_idx, region in genome.output_regions():
        assert(region == [-2, 0, None])


# -> genome
def test_check_dna_consistency():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 0, None]

    # invalid length
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, -2, -1, -2, 0, None, 0]

    # invalid function gene for input node
    with pytest.raises(ValueError):
        genome.dna = [0, None, None, -1, None, None, 0, -2, 0, -2, 0, None]

    # invalid input gene for input node
    with pytest.raises(ValueError):
        genome.dna = [-1, 0, None, -1, None, None, 0, -2, 0, -2, 0, None]

    # invalid function gene for hidden node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 2, 0, 1, -2, 0, None]

    # invalid input gene for hidden node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, 2, 1, -2, 0, None]

    # invalid function gene for output node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 0, 0, None]

    # invalid input gene for input node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 3, None]

    # invalid non-coding input gene for output node
    with pytest.raises(ValueError):
        genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 0, 0]


# -> node
def test_add():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert(abs(x[0] + x[1] - y[0]) < 1e-15)


# -> node
def test_sub():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPSub])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert(abs(x[0] - x[1] - y[0]) < 1e-15)


# -> node
def test_mul():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 1,
        'n_rows': 1,
        'levels_back': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPMul])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)

    x = [5., 1.5]
    y = graph(x)

    assert(abs((x[0] * x[1]) - y[0]) < 1e-15)


# -> graph
def test_direct_input_output():
    params = {
        'n_inputs': 1,
        'n_outputs': 1,
        'n_columns': 3,
        'n_rows': 3,
        'levels_back': 2,
    }
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.randomize()

    genome[-2:] = [0, None]  # set inputs for output node to input node
    graph = gp.CGPGraph(genome)

    x = [2.14159]
    y = graph(x)

    assert(abs(x[0] - y[0]) < 1e-15)


# -> primitives
def test_immutable_primitives():
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    with pytest.raises(TypeError):
        primitives[0] = gp.CGPAdd

    # currently setting this possible, since MappingProxy which was
    # used to enforce this behaviour can not be pickled and hence was
    # removed from Primitives
    # with pytest.raises(TypeError):
    #     primitives._primitives[0] = gp.CGPAdd


# -> primitives
def test_max_arity():
    plain_primitives = [gp.CGPAdd, gp.CGPSub, gp.CGPConstantFloat]
    primitives = gp.CGPPrimitives(plain_primitives)

    arity = 0
    for p in plain_primitives:
        if arity < p._arity:
            arity = p._arity

    assert(arity == primitives.max_arity)


# -> graph
def test_compile_simple():
    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(2, 1, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(x[0] + x[1] - y[0]) < 1e-15)

    primitives = gp.CGPPrimitives([gp.CGPSub])
    genome = gp.CGPGenome(2, 1, 1, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, -2, 2, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(x[0] - x[1] - y[0]) < 1e-15)


# -> graph
def test_compile_two_columns():
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    genome = gp.CGPGenome(2, 1, 2, 1, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 1, 0, 2, -2, 3, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(x[0] - (x[0] + x[1]) - y[0]) < 1e-15)


# -> graph
def test_compile_two_columns_two_rows():
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    genome = gp.CGPGenome(2, 2, 2, 2, 1, primitives)

    genome.dna = [-1, None, None, -1, None, None, 0, 0, 1, 1, 0, 1, 0, 0, 2, 0, 2, 3, -2, 4, None, -2, 5, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(x[0] + (x[0] + x[1]) - y[0]) < 1e-15)
    assert(abs((x[0] + x[1]) + (x[0] - x[1]) - y[1]) < 1e-15)


# -> node
def test_compile_addsubmul():
    params = {
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 2,
        'n_rows': 2,
        'levels_back': 1,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul])
    genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)
    genome.dna = [
        -1, None, None, -1, None, None,
        2, 0, 1, 1, 0, 1,
        1, 2, 3, 0, 0, 0,
        -2, 4, None]
    graph = gp.CGPGraph(genome)
    f = graph.compile_func()

    x = [5., 2.]
    y = f(x)

    assert(abs(((x[0] * x[1]) - (x[0] - x[1])) - y[0]) < 1e-15)


# -> graph
def test_compile_torch_and_backprop():
    primitives = gp.CGPPrimitives([gp.CGPMul, gp.CGPConstantFloat])
    genome = gp.CGPGenome(1, 1, 2, 2, 1, primitives)
    genome.dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 3, None]
    graph = gp.CGPGraph(genome)

    c = graph.compile_torch_class()

    optimizer = torch.optim.SGD(c.parameters(), lr=1e-1)
    criterion = torch.nn.MSELoss()

    for i in range(200):

        x = torch.Tensor(1, 1).normal_()
        y = c(x)

        y_target = -2.14159 * x

        loss = criterion(y, y_target)
        c.zero_grad()
        loss.backward()

        optimizer.step()

    assert(loss.detach().numpy() < 1e-15)

    x = [3.]
    x_torch = torch.Tensor(x).view(1, 1)
    assert(abs(c(x_torch)[0].detach().numpy() - graph(x))[0] > 1e-15)
    graph.update_parameters_from_torch_class(c)
    assert(abs(c(x_torch)[0].detach().numpy() - graph(x))[0] < 1e-15)


def test_compile_sympy_expression():
    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPConstantFloat])
    genome = gp.CGPGenome(1, 1, 2, 2, 1, primitives)

    genome.dna = [-1, None, None, 1, None, None, 1, None, None, 0, 0, 1, 0, 0, 1, -2, 3, None]
    graph = gp.CGPGraph(genome)

    x_0_target, y_0_target = sympy.symbols('x_0_target y_0_target')
    y_0_target = x_0_target + 1.

    y_0 = graph.compile_sympy_expression()[0]

    for x in np.random.normal(size=100):
        assert abs(y_0_target.subs('x_0_target', x).evalf() - y_0.subs('x_0', x).evalf()) < 1e-12


def test_catch_no_non_coding_allele_in_non_coding_region():
    primitives = gp.CGPPrimitives([gp.CGPConstantFloat])
    genome = gp.CGPGenome(1, 1, 1, 1, 1, primitives)

    # wrong: ConstantFloat node has no inputs, but input gene has
    # value different from the non-coding allele
    with pytest.raises(ValueError):
        genome.dna = [-1, None, 0, 0, -2, 1]

    # correct
    genome.dna = [-1, None, 0, None, -2, 1]


def test_individuals_have_different_genomes():

    params = {
        'seed': 8188212,

        # evo parameters
        'n_parents': 5,
        'n_offspring': 5,
        'generations': 50000,
        'n_breeding': 5,
        'tournament_size': 2,
        'mutation_rate': 0.05,

        # cgp parameters
        'n_inputs': 2,
        'n_outputs': 1,
        'n_columns': 6,
        'n_rows': 6,
        'levels_back': 2,
    }

    primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPDiv, gp.CGPConstantFloat])

    pop = gp.CGPPopulation(
        params['n_parents'], params['n_offspring'], params['n_breeding'], params['tournament_size'], params['mutation_rate'],
        params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], params['levels_back'], primitives)

    pop.generate_random_parent_population()
    pop.generate_random_offspring_population()

    for i, ind in enumerate(pop):
        ind.fitness = -i

    pop.create_combined_population()

    for i, ind in enumerate(pop._combined):
        ind.fitness = -i

    pop.sort()

    pop.create_new_parent_population()
    pop.create_new_offspring_population()

    for i, parent_i in enumerate(pop._parents):

        for j, parent_j in enumerate(pop._parents):
            if i != j:
                assert parent_i is not parent_j
                assert parent_i.genome is not parent_j.genome
                assert parent_i.genome.dna is not parent_j.genome.dna

        for j, offspring_j in enumerate(pop._offsprings):
            if i != j:
                assert parent_i is not offspring_j
                assert parent_i.genome is not offspring_j.genome
                assert parent_i.genome.dna is not offspring_j.genome.dna


def test_pickle_individual():

    primitives = gp.CGPPrimitives([gp.CGPAdd])
    genome = gp.CGPGenome(1, 1, 1, 1, 1, primitives)
    individual = gp.CGPIndividual(None, genome)

    with open('individual.pkl', 'wb') as f:
        pickle.dump(individual, f)

# def cgp():
#     params = {
#         'seed': 81882,
#         'n_inputs': 2,
#         'n_outputs': 1,
#         'n_columns': 3,
#         'n_rows': 3,
#         'levels_back': 2,
#         'mutation_rate': 0.05,
#     }

#     np.random.seed(params['seed'])

#     primitives = gp.CGPPrimitives([gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPConstantFloat])
#     genome = gp.CGPGenome(params['n_inputs'], params['n_outputs'], params['n_columns'], params['n_rows'], primitives)
#     genome.randomize(params['levels_back'])
#     graph = gp.CGPGraph(genome)

#     history_loss = []
#     for i in range(3000):

#         genome.mutate(params['mutation_rate'], params['levels_back'])
#         graph.parse_genome(genome)
#         f = graph.compile_torch_class()

#         if len(list(f.parameters())) > 0:
#             optimizer = torch.optim.SGD(f.parameters(), lr=1e-1)
#             criterion = torch.nn.MSELoss()

#         history_loss_trial = []
#         history_loss_bp = []
#         for j in range(100):
#             x = torch.Tensor(2).normal_()
#             y = f(x)
#             loss = (2.7182 + x[0] - x[1] - y[0]) ** 2
#             history_loss_trial.append(loss.detach().numpy())

#             if len(list(f.parameters())) > 0:
#                 y_target = 2.7182 + x[0] - x[1]

#                 loss = criterion(y[0], y_target)
#                 f.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 history_loss_bp.append(loss.detach().numpy())

#         graph.update_parameter_values(f)

#         if np.mean(history_loss_trial[-10:]) < 1e-1:
#             print(graph[-1].output_str)
#             print(graph)

#             if len(list(f.parameters())) > 0:
#                 plt.plot(history_loss_bp)
#                 plt.show()

#         history_loss.append(np.sum(np.mean(history_loss_trial)))

#     plt.plot(history_loss)
#     plt.show()
