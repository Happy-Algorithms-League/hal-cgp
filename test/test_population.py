import copy
import gp
import pytest
import numpy as np


def test_assert_mutation_rate(rng_seed, genome_params, mutation_rate):
    with pytest.raises(ValueError):
        gp.Population(5, -0.1, rng_seed, genome_params)

    with pytest.raises(ValueError):
        gp.Population(5, 1.1, rng_seed, genome_params)

    # assert that no error is thrown for a suitable mutation rate
    gp.Population(5, mutation_rate, rng_seed, genome_params)


def test_init_random_parent_population(population_params, genome_params):
    pop = gp.Population(**population_params, genome_params=genome_params)
    assert len(pop.parents) == population_params["n_parents"]


def test_champion(population_simple_fitness):
    pop = population_simple_fitness
    assert pop.champion == pop.parents[-1]


def test_crossover_too_small(population_simple_fitness):
    pop = population_simple_fitness
    # Breeding pool too small
    with pytest.raises(AssertionError):
        pop.crossover(pop.parents[:1], 2)


def test_crossover_one_offspring_all_parents(population_simple_fitness):
    pop = population_simple_fitness
    # Check is best parent is chosen if n_offsprings = 1
    offspring = pop.crossover(pop.parents, 1)
    assert offspring[0] == pop.champion


def test_crossover_one_offspring_breeding_pool(population_simple_fitness):
    pop = population_simple_fitness
    # Check is best parent in smaller breeding pool is chosen if n_offsprings = 1
    offspring = pop.crossover(pop.parents[:3], 1)
    assert offspring[0] == max(pop.parents[:3], key=lambda x: x.fitness)


def test_crossover_two_offspring(population_simple_fitness):
    pop = population_simple_fitness
    # Check if best two parents are chosen if n_offsprings = 2
    offspring = pop.crossover(pop.parents, 2)
    assert offspring[0] == pop.champion
    assert offspring[1] == sorted(pop.parents, key=lambda x: -x.fitness)[1]


def test_mutate(population_params, genome_params):
    population_params["mutation_rate"] = 0.999999
    pop = gp.Population(**population_params, genome_params=genome_params)

    offspring = pop.parents
    offspring_original = copy.deepcopy(offspring)
    offspring = pop.mutate(offspring)
    assert np.any(
        [off_orig != off_mutated for off_orig, off_mutated in zip(offspring_original, offspring)]
    )


def test_fitness_parents(population_params, genome_params):
    pop = gp.Population(**population_params, genome_params=genome_params)
    fitness_values = np.random.rand(population_params["n_parents"])
    for fitness, parent in zip(fitness_values, pop.parents):
        parent.fitness = fitness

    assert np.all(pop.fitness_parents() == fitness_values)
