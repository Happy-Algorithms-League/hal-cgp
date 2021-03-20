import copy
import functools

import numpy as np
import pytest

import cgp


def test_assert_mutation_rate(n_offsprings, mutation_rate):
    with pytest.raises(ValueError):
        cgp.ea.MuPlusLambda(n_offsprings, -0.1)

    with pytest.raises(ValueError):
        cgp.ea.MuPlusLambda(n_offsprings, 1.1)

    # assert that no error is thrown for a suitable mutation rate
    cgp.ea.MuPlusLambda(n_offsprings, mutation_rate)


def test_objective_with_label(population_params, genome_params, ea_params):
    def objective_without_label(individual):
        individual.fitness = -2.0
        return individual

    def objective_with_label(individual, label):
        assert label == "test"
        individual.fitness = -1.0
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)
    ea.initialize_fitness_parents(pop, objective_without_label)

    ea.step(pop, objective_without_label)
    assert pop.champion.fitness == pytest.approx(-2.0)

    obj = functools.partial(objective_with_label, label="test")
    ea.step(pop, obj)
    assert pop.champion.fitness == pytest.approx(-1.0)


def test_fitness_contains_and_maintains_nan(population_params, genome_params, ea_params):
    def objective(individual):
        if np.random.rand() < 0.95:
            individual.fitness = np.nan
        else:
            individual.fitness = np.random.rand()
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)
    ea.initialize_fitness_parents(pop, objective)
    ea.step(pop, objective)
    assert any([np.isnan(ind.fitness) for ind in pop])


def test_offspring_individuals_are_assigned_correct_indices(
    population_params, genome_params, ea_params
):
    def objective(ind):
        ind.fitness = 0.0
        return ind

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)
    ea.initialize_fitness_parents(pop, objective)

    offsprings = ea._create_new_offspring_generation(pop)

    for idx, ind in enumerate(offsprings):
        assert ind.idx == len(pop.parents) + idx


def test_offspring_individuals_are_assigned_correct_parent_indices(
    population_params, genome_params, ea_params
):
    def objective(ind):
        ind.fitness = 0.0
        return ind

    population_params["n_parents"] = 1
    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea_params["tournament_size"] = 1
    ea = cgp.ea.MuPlusLambda(**ea_params)
    ea.initialize_fitness_parents(pop, objective)

    offsprings = ea._create_new_offspring_generation(pop)

    for ind in offsprings:
        assert ind.parent_idx == 0


def test_local_search_is_only_applied_to_best_k_individuals(
    population_params, local_search_params, ea_params
):

    torch = pytest.importorskip("torch")

    def inner_objective(f):
        return torch.nn.MSELoss()(
            torch.DoubleTensor([[1.1]]), f(torch.zeros(1, 1, dtype=torch.double))
        )

    def objective(ind):
        if not ind.fitness_is_None():
            return ind

        f = ind.to_torch()
        ind.fitness = -inner_objective(f).item()
        return ind

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
        cgp.local_search.gradient_based, objective=inner_objective, **local_search_params
    )

    ea = cgp.ea.MuPlusLambda(**ea_params, local_search=local_search, k_local_search=k_local_search)
    ea.initialize_fitness_parents(pop, objective)
    ea.step(pop, objective)

    for idx in range(k_local_search):
        assert pop[idx].genome._parameter_names_to_values["<p1>"] != pytest.approx(1.0)

    for idx in range(k_local_search, population_params["n_parents"]):
        assert pop[idx].genome._parameter_names_to_values["<p1>"] == pytest.approx(1.0)


def test_raise_fitness_has_wrong_type(population_params, genome_params, ea_params):
    def objective(individual):
        individual.fitness = int(5.0)  # should raise error since fitness should be float
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    with pytest.raises(ValueError):
        ea.initialize_fitness_parents(pop, objective)


def test_initialize_fitness_parents(population_params, genome_params, ea_params):
    def objective(individual):
        individual.fitness = -1.0
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)
    ea.initialize_fitness_parents(pop, objective)
    assert all([not ind.fitness_is_None() for ind in pop.parents])


def test_step(population_params, genome_params, ea_params):
    def objective(individual):
        individual.fitness = float(individual.idx)
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)
    ea.initialize_fitness_parents(pop, objective)
    old_parent_ids = sorted([ind.idx for ind in pop.parents])
    ea.step(pop, objective)
    new_parent_ids = sorted([ind.idx for ind in pop.parents])
    # After one step, the new parent population should have IDs that
    # are offset from the old parent ids by n_offsprings
    # This is by construction in this test because the fitness is equal to the id
    assert all(
        [
            new_id == old_id + ea_params["n_offsprings"]
            for new_id, old_id in zip(new_parent_ids, old_parent_ids)
        ]
    )


def test_sort(population_params, genome_params, ea_params):
    def objective(individual):
        individual.fitness = float(individual.idx)
        return individual

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)
    ea.initialize_fitness_parents(pop, objective)
    sorted_parents = ea._sort(pop.parents)
    # Assert that the sorting inverted the list of parents (because the fitness is equal to the id)
    assert sorted_parents == pop.parents[::-1]


def test_create_new_offspring_and_parent_generation(population_params, genome_params, ea_params):
    def objective(individual):
        individual.fitness = float(individual.idx)
        return individual

    ea_params["mutation_rate"] = 1.0  # ensures every offspring has mutations

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    ea.initialize_fitness_parents(pop, objective)

    offsprings = ea._create_new_offspring_generation(pop)
    assert len(offsprings) == ea_params["n_offsprings"]
    assert all([ind.idx >= pop.n_parents for ind in offsprings])
    # Assert that all offspring dna are different from all parents dna
    offspring_dna = [ind.genome.dna for ind in offsprings]
    parent_dna = [ind.genome.dna for ind in pop.parents]
    assert all([odna != pdna for odna in offspring_dna for pdna in parent_dna])


def test_create_new_parent_population(population_params, genome_params, ea_params):
    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    # Create new parent population from the parents and assert that
    # we picked the first three individuals
    new_parents = ea._create_new_parent_population(3, pop.parents)
    assert new_parents == pop.parents[:3]


def test_update_n_objective_calls(population_params, genome_params, ea_params):
    def objective(individual):
        individual.fitness = float(individual.idx)
        return individual

    n_objective_calls_expected = 0
    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)
    assert ea.n_objective_calls == n_objective_calls_expected

    ea.initialize_fitness_parents(pop, objective)
    n_objective_calls_expected = population_params["n_parents"]
    assert ea.n_objective_calls == n_objective_calls_expected

    n_generations = 100
    for _ in range(n_generations):
        offsprings = ea._create_new_offspring_generation(pop)
        combined = offsprings + pop.parents
        n_objective_calls_expected += sum([1 for ind in combined if ind.fitness_is_None()])
        combined = ea._compute_fitness(combined, objective)
        assert n_objective_calls_expected == ea.n_objective_calls


def test_update_n_objective_calls_mutation_rate_one(population_params, genome_params, ea_params):
    def objective(individual):
        individual.fitness = float(individual.idx)
        return individual

    ea_params["mutation_rate"] = 1.0
    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)
    ea.initialize_fitness_parents(pop, objective)
    n_objective_calls_expected = population_params["n_parents"]
    n_step_calls = 100
    for idx_current_step in range(n_step_calls):
        ea.step(pop, objective)
        n_objective_calls_expected += ea_params["n_offsprings"]
        assert ea.n_objective_calls == n_objective_calls_expected


def test_hurdles(population_params, genome_params, ea_params):

    # make sure all offsprings are assigned fitness None
    population_params["n_parents"] = 3
    ea_params["mutation_rate"] = 1.0
    ea_params["n_offsprings"] = 3
    ea_params["hurdle_percentile"] = [0.1, 0.0]

    def objective_one(ind):
        # assign low fitness to individuals 4 and 5 to check blocking via hurdle
        if ind.idx in (4, 5):
            ind.fitness = -float(ind.idx)
        else:
            ind.fitness = float(ind.idx)
        return ind

    def objective_two(ind):
        if ind.idx == 4:
            ind.fitness = -((1.0 + float(ind.idx)) ** 2)
        else:
            ind.fitness = (1.0 + float(ind.idx)) ** 2
        return ind

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    ea.initialize_fitness_parents(pop, [objective_one, objective_two])

    # while initializing parents, both objectives should have been
    # evaluated for all parents; the parents fitness is hence the sum
    # of both objectives
    parents_expected = [(0, 1), (1, 5), (2, 11)]
    for ind, ind_expected in zip(pop.parents, parents_expected):
        assert ind.idx == ind_expected[0]
        assert ind.fitness == pytest.approx(ind_expected[1])

    # code below implements `ea.step`, but keeps offsprings around to
    # check combined population
    offsprings = ea._create_new_offspring_generation(pop)
    combined = offsprings + pop.parents

    combined = ea._compute_fitness(combined, [objective_one, objective_two])
    combined = ea._sort(combined)

    # individual 4 has higher fitness as individual 5 as the latter
    # didn't make it past the first hurdle
    combined_expected = [(3, 19), (2, 11), (1, 5), (0, 1), (4, -29), (5, -5)]
    assert len(combined) == len(combined_expected)
    for ind, ind_expected in zip(combined, combined_expected):
        assert ind.idx == ind_expected[0]
        assert ind.fitness == pytest.approx(ind_expected[1])


def test_mutate(population_params, genome_params, ea_params):
    ea_params["mutation_rate"] = 0.5
    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    offspring = pop.parents
    offspring_original = copy.deepcopy(offspring)
    offspring = ea.mutate(offspring, pop.rng)
    assert np.any(
        [off_orig != off_mutated for off_orig, off_mutated in zip(offspring_original, offspring)]
    )
