"""
Example: Solving an OpenAI Gym environment with CGP.
====================================================

This examples demonstrates how to solve an OpenAI Gym environment
(https://gym.openai.com/envs/) with Cartesian genetic programming. We
choose the "MountainCarContinuous" environment due to its continuous
observation and action spaces.

Preparatory steps:
Install the OpenAI Gym package: `pip install gym`

"""


import functools

try:
    import gym
except ImportError:
    raise ImportError(
        "Failed to import the OpenAI Gym package. Please install it via `pip install gym`."
    )

import matplotlib.pyplot as plt
import numpy as np
import warnings

import cgp

# %%
# For more flexibility in the evolved expressions, we define two
# constants that can be used in the expressions, with values 0.1 and
# 10.


class ConstantFloatZeroPointOne(cgp.ConstantFloat):
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 0.1


class ConstantFloatTen(cgp.ConstantFloat):
    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)
        self._output = 10.0


# %%
# Then we define the objective function for the evolution.  The inner
# objective accepts a Python callable as input. This callable
# determines the action taken by the agent upon receiving observations
# from the environment. The fitness of the given callable on the task
# is then computed as the cumulative reward over a fixed number of
# episodes.


def inner_objective(f, seed, n_total_steps, *, render):

    np.random.seed(seed)

    env = gym.make("MountainCarContinuous-v0")

    observation = env.reset()
    env.seed(seed)

    cum_reward_all_episodes = []
    cum_reward_this_episode = 0
    for _ in range(n_total_steps):

        if render:
            env.render()

        continuous_action = f(observation)
        observation, reward, done, _ = env.step(continuous_action)
        cum_reward_this_episode += reward

        if done:
            cum_reward_all_episodes.append(cum_reward_this_episode)
            cum_reward_this_episode = 0
            observation = env.reset()

    env.close()

    return cum_reward_all_episodes


# %%
# The objective then takes an individual, evaluates the inner
# objective, and updates the fitness of the individual. If the
# expression of the individual leads to a division by zero, this error
# is caught and the individual gets a fitness of -infinity assigned.


def objective(ind, seed, n_total_steps):

    if ind.fitness is not None:
        return ind

    f = ind.to_func()
    try:
        with warnings.catch_warnings():  # ignore warnings due to zero division
            warnings.filterwarnings(
                "ignore", message="divide by zero encountered in double_scalars"
            )
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in double_scalars"
            )
            cum_reward_all_episodes = inner_objective(f, seed, n_total_steps, render=False)

        # more episodes are better, more reward is better
        n_episodes = float(len(cum_reward_all_episodes))
        mean_cum_reward = np.mean(cum_reward_all_episodes)
        ind.fitness = n_episodes + 1.0 * mean_cum_reward

    except ZeroDivisionError:
        ind.fitness = -np.inf

    return ind


# %%
# We then define the main loop for the evolution, which consists of:
#
# - parameters for the population, the genome of individuals, and the evolutionary algorithm.
# - creating a Population instance and instantiating the evolutionary algorithm.
# - defining a recording callback closure for bookkeeping of the progression of the evolution.
#
# Finally, we call the `evolve` method to perform the evolutionary search.


def evolve(seed):

    objective_params = {"n_total_steps": 2000}

    population_params = {"n_parents": 1, "mutation_rate": 0.04, "seed": seed}

    genome_params = {
        "n_inputs": 2,
        "n_outputs": 1,
        "n_columns": 16,
        "n_rows": 1,
        "levels_back": None,
        "primitives": (
            cgp.Add,
            cgp.Sub,
            cgp.Mul,
            cgp.Div,
            cgp.ConstantFloat,
            ConstantFloatZeroPointOne,
            ConstantFloatTen,
        ),
    }

    ea_params = {"n_offsprings": 4, "tournament_size": 1, "n_processes": 4}

    evolve_params = {"max_generations": 3000, "min_fitness": 200.0}

    pop = cgp.Population(**population_params, genome_params=genome_params)

    ea = cgp.ea.MuPlusLambda(**ea_params)

    history = {}
    history["expr_champion"] = []
    history["fitness_champion"] = []

    def recording_callback(pop):
        history["expr_champion"].append(pop.champion.to_sympy())
        history["fitness_champion"].append(pop.champion.fitness)

    obj = functools.partial(objective, seed=seed, n_total_steps=objective_params["n_total_steps"])

    cgp.evolve(pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback)

    return history


# %%
# For visualization, we define a function to plot the fitness over generations.


def plot_fitness_over_generation_index(history):
    width = 6.0
    fig = plt.figure(figsize=(width, width / 1.618))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax.set_xlabel("Generation index")
    ax.set_ylabel("Fitness champion")
    ax.plot(history["fitness_champion"])
    fig.savefig("example_mountain_car.png", dpi=300)


# %%
# We define a function that checks whether the best expression
# fulfills the "solving criteria", i.e., average reward of at least
# 90.0 over 100 consecutive
# trials. (https://github.com/openai/gym/wiki/Leaderboard#mountaincarcontinuous-v0)


def evaluate_best_expr(expr):

    np.random.seed(seed)

    env = gym.make("MountainCarContinuous-v0")

    observation = env.reset()
    env.seed(seed)

    def f(x):
        res = [float(expr.subs({"x_0": x[0], "x_1": x[1]}).evalf())]
        return res

    cum_reward_all_episodes = []
    cum_reward_this_episode = 0
    while len(cum_reward_all_episodes) < 100:

        continuous_action = f(observation)
        observation, reward, done, _ = env.step(continuous_action)
        cum_reward_this_episode += reward

        if done:
            cum_reward_all_episodes.append(cum_reward_this_episode)
            cum_reward_this_episode = 0
            observation = env.reset()

    env.close()

    cum_reward_average = np.mean(cum_reward_all_episodes)
    print(f"average reward over 100 consecutive trials: {cum_reward_average:.05f}", end="")
    if cum_reward_average >= 90.0:
        print("-> environment solved!")
    else:
        print()

    return cum_reward_all_episodes


# %%
# Furthermore, we define a function for visualizing the agent's behaviour for
# each expression that increase over the currently best performing individual.


def visualize_behaviour_for_evolutionary_jumps(seed, history, only_final_solution=True):
    n_total_steps = 999

    max_fitness = -np.inf
    for i, fitness in enumerate(history["fitness_champion"]):

        if only_final_solution and i != (len(history["fitness_champion"]) - 1):
            continue

        if fitness > max_fitness:
            expr = history["expr_champion"][i][0]
            expr_str = str(expr).replace("x_0", "x").replace("x_1", "dx/dt x")

            print(f'visualizing behaviour for expression "{expr_str}" (fitness: {fitness:.05f})')

            def f(x):
                res = [float(expr.subs({"x_0": x[0], "x_1": x[1]}).evalf())]
                return res

            inner_objective(f, seed, n_total_steps, render=True)

            max_fitness = fitness


# %%
# Finally, we execute the evolution and visualize the results.


if __name__ == "__main__":

    seed = 818821

    print("starting evolution")
    history = evolve(seed)
    print("evolution ended")

    max_fitness = history["fitness_champion"][-1]
    best_expr = history["expr_champion"][-1][0]
    best_expr_str = str(best_expr).replace("x_0", "x").replace("x_1", "dx/dt x")
    print(f'solution with highest fitness: "{best_expr_str}" (fitness: {max_fitness:.05f})')

    plot_fitness_over_generation_index(history)
    evaluate_best_expr(best_expr)
    visualize_behaviour_for_evolutionary_jumps(seed, history)
