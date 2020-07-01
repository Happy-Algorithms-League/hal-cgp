import functools
import numpy as np
import pytest

import cgp


def test_objective_with_label(population_params, genome_params):
    def objective_without_label(individual):
        individual.fitness = -2.0
        return individual

    def objective_with_label(individual, label):
        assert label == "test"
        individual.fitness = -1.0
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(1, 2, 1)
    ea.initialize_fitness_parents(pop, objective_without_label)

    ea.step(pop, objective_without_label)
    assert pop.champion.fitness == pytest.approx(-2.0)

    obj = functools.partial(objective_with_label, label="test")
    ea.step(pop, obj)
    assert pop.champion.fitness == pytest.approx(-1.0)


def test_fitness_contains_and_maintains_nan(population_params, genome_params):
    def objective(individual):
        if np.random.rand() < 0.95:
            individual.fitness = np.nan
        else:
            individual.fitness = np.random.rand()
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(10, 10, 1)
    ea.initialize_fitness_parents(pop, objective)
    ea.step(pop, objective)

    assert np.nan in [ind.fitness for ind in pop]


def test_offspring_individuals_are_assigned_correct_indices(population_params, genome_params):
    def objective(ind):
        ind.fitness = 0.0
        return ind

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(10, 10, 1)
    ea.initialize_fitness_parents(pop, objective)

    offsprings = ea._create_new_offspring_generation(pop)

    for idx, ind in enumerate(offsprings):
        assert ind.idx == len(pop.parents) + idx


def test_local_search_is_only_applied_to_best_k_individuals(
    population_params, local_search_params
):

    torch = pytest.importorskip("torch")

    def inner_objective(f):
        return torch.nn.MSELoss()(torch.Tensor([[1.1]]), f(torch.zeros(1, 1)))

    def objective(ind):
        if ind.fitness is not None:
            return ind

        f = ind.to_torch()
        ind.fitness = -inner_objective(f).item()
        return ind

    population_params["mutation_rate"] = 0.3

    genome_params = {
        "n_inputs": 1,
        "n_outputs": 1,
        "n_columns": 1,
        "n_rows": 1,
        "levels_back": None,
        "primitives": (cgp.Parameter,),
    }

    k_local_search = 2

    pop = cgp.Population(**population_params, genome_params=genome_params)

    local_search = functools.partial(
        cgp.local_search.gradient_based, objective=inner_objective, **local_search_params,
    )

    ea = cgp.ea.MuPlusLambda(5, 5, 1, local_search=local_search, k_local_search=k_local_search)
    ea.initialize_fitness_parents(pop, objective)
    ea.step(pop, objective)

    for idx in range(k_local_search):
        assert pop[idx].genome._parameter_names_to_values["<p1>"] != pytest.approx(1.0)

    for idx in range(k_local_search, population_params["n_parents"]):
        assert pop[idx].genome._parameter_names_to_values["<p1>"] == pytest.approx(1.0)
