import numpy as np
from typing import Callable


class RandomLocalSearch:
    def __init__(
        self,
        objective: Callable,
        seed: int,
        n_steps: int = 10,  # todo: should this be dependent on the number of parameters?
        permissible_values: np.ndarray = np.logspace(start=0, stop=7, num=8, base=2, dtype=int),
    ) -> None:
        self.objective = objective
        self.seed = seed
        self.n_steps = n_steps
        self.permissible_values = permissible_values

    def __call__(self, ind) -> None:
        rng = np.random.RandomState(self.seed)

        params_values, params_names = ind.parameters_to_numpy_array(only_active_nodes=True)

        if len(params_values) > 0:
            for _ in range(self.n_steps):
                # sample a new set of parameter values randomly
                params_sampled = [
                    rng.choice(self.permissible_values) for param_value in params_values
                ]
                # write the parameters into a clone of individual
                new_ind = ind.clone()
                new_ind.update_parameters_from_numpy_array(
                    params=params_sampled, params_names=params_names
                )
                # evaluate fitness
                self.objective(new_ind)
                # if fitness improved: replace parameter values and fitness
                if new_ind.fitness >= ind.fitness:  # todo: should this be >= or > ??
                    ind.update_parameters_from_numpy_array(
                        params=params_sampled, params_names=params_names
                    )
                    ind.fitness = new_ind.fitness


if __name__ == "__main__":
    import cgp

    def objective(ind):
        params_values, _ = ind.parameters_to_numpy_array(only_active_nodes=True)
        ind.fitness = np.sum(params_values)
        return ind

    seed = 12345
    genome = cgp.Genome(primitives=(cgp.Add, cgp.Sub, cgp.Mul, cgp.Parameter), n_inputs=1)
    genome.randomize(rng=np.random.RandomState(seed=seed))
    ind = cgp.IndividualSingleGenome(genome=genome)

    objective(ind)
    print(
        f"Node parameters before local search "
        f"{ind.parameters_to_numpy_array(only_active_nodes=True)} \n"
    )

    rls = RandomLocalSearch(objective=objective, seed=seed, n_steps=1000)
    rls(ind)
    print(
        f"Node parameters after local search "
        f"{ind.parameters_to_numpy_array(only_active_nodes=True)} \n"
    )
