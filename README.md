python-gp
=========
[![Python3.6](https://img.shields.io/badge/python-3.6-red.svg)](https://www.python.org/downloads/release/python-369/)
[![Python3.7](https://img.shields.io/badge/python-3.7-red.svg)](https://www.python.org/)
[![Python3.8](https://img.shields.io/badge/python-3.8-red.svg)](https://www.python.org/)
[![GPL license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-3.0.html)

Cartesian Genetic Programming (CGP) in Python.

This library implements Cartesian Genetic Programming (e.g, Miller and Thomson, 2000; Miller, 2011) for symbolic regression in pure Python, targeting applications with expensive fitness evaluations. It provides Python data structures to represent and evolve two-dimensional directed graphs (genotype) that are translated into computational graphs (phenotype) implementing mathematical expressions. The computational graphs can be compiled as a Python functions, SymPy expressions (Meurer et al., 2017) or PyTorch modules (Paszke et al., 2017). The library currently implements an evolutionary algorithm, specifically (mu + lambda) evolution strategies adapted from Deb et al. (2002), to evolve a population of symbolic expressions in order to optimize an objective function.

<div style="text-align:center"><img src="cgp-sketch.png" alt="CGP Sketch" width="600"/></div>

A simple example of CGP applied to a symbolic regression problem can be found in `examples/example_evo_regression.py`.


Basic usage
===========

Follow these steps to solve a basic regression problem:

1. Define an objective function. 

   The objective function takes an individual as an argument and updates the `fitness` of the individual.
```python
def objective(individual):
      """Objective function of the regression task.

       Parameters
      ----------
      individual : gp.Individual
          Individual of the Cartesian Genetic Programming Framework.

      Returns
      -------
      gp.Individual
          Modified individual with updated fitness value.
      """
      # Compute the fitness value
      individual.fitness = ...
      return individual
```
2. Define parameters for the genome, the population and the evolutionary algorithm
```python
params = {
     "seed": 8188212,
     "max_generations": 1000,
     "min_fitness": 0.0,

     "population_params": {
          "n_parents": 10,
          "mutation_rate": 0.5,
      },

     "ea_params": {
         "n_offsprings": 10,
         "n_breeding": 10,
         "tournament_size": 1,
         "n_processes": 2,
      },

     "genome_params": {
         "n_inputs": 2,
         "n_outputs": 1,
         "n_columns": 10,
         "n_rows": 5,
         "levels_back": 2,
         "primitives": [gp.Add, gp.Sub, gp.Mul, gp.Div, gp.ConstantFloat]},
      }
```
3. Initialize the population and an evolutionary algorithm instance:
```python
pop = gp.Population(**params["population_params"],
                    seed=params["seed"], genome_params=params["genome_params"])
ea = gp.ea.MuPlusLambda(**params["ea_params"])
```
4. Define a callback function to record information about the evolution of the population:
```python
history = {}
history["fitness_parents"] = []
def recording_callback(pop):
    history["fitness_parents"].append(pop.fitness_parents())
```
5. Use the `evolve` function from the high-level API that ties everything together and executes the evolution:
```python
history = gp.evolve(pop, obj, ea, params["max_generations"],
                    params["min_fitness"], print_progress=True, callback=recording_callback)
```


References
==========

Miller, J. and Thomson, P. (2000). Cartesian Genetic Programming. In Proc. European Conference on Genetic Programming, volume 1802, pages 121–132. Springer.

Miller, J. F. (2011). Cartesian Genetic Programming. In Cartesian Genetic Programming, pages 17–34. Springer.

Meurer, A., Smith, C. P., Paprocki, M., Čertík, O., Kirpichev, S. B., Rocklin, M., ... & Rathnayake, T. (2017). SymPy: Symbolic Computing in Python. PeerJ Computer Science, 3, e103.

Paszke, A., Gross, S., Chintala, S., Chanan, G., Yang, E., DeVito, Z., ... & Lerer, A. (2017). Automatic Differentiation in PyTorch.

Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE transactions on evolutionary computation, 6(2), 182-197.


Code status
===========

[![Build Status](https://travis-ci.org/jakobj/python-gp.svg?branch=master)](https://travis-ci.org/jakobj/python-gp)
