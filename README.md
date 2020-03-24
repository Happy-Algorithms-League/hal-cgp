python-gp
===========
[![Python3.6](https://img.shields.io/badge/python-3.6-red.svg)](https://www.python.org/downloads/release/python-369/)
[![Python3.7](https://img.shields.io/badge/python-3.7-red.svg)](https://www.python.org/)
[![Python3.8](https://img.shields.io/badge/python-3.8-red.svg)](https://www.python.org/)
[![GPL license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-3.0.html)

Cartesian Genetic Programming (CGP) in Python.

This library implements Cartesian Genetic Programming for symbolic regression in pure Python. It provides Python data structures to represent and evolve a two-dimensional directed graph (the genotype) that can be translated into a computational graph (the phenotype) implementing a particular mathematical expression. The current implementation uses an evolutionary algorithm,  specifically (mu + lambda) evolution strategies, to evolve a population of symbolic expressions in order to optimize an objective function.

<img src="cgp-sketch.png" alt="CGP Sketch" width="600"/>

Basic usage
===========

A simple example of CGP applied to a symbolic regression problem can be found in `examples/example_evo_regression.py`.

The basic steps are:

1. Define an objective function. 

   The objective function takes an individual as an argument and updates the `fitness` of the individual.
```python
def objective(individual):
      """Objective function of the regression task.

       Parameters
      ----------
      individual : gp.CGPIndividual
          Individual of the Cartesian Genetic Programming Framework.

      Returns
      -------
      gp.CGPIndividual
          Modified individual with updated fitness value.
      """
      # Compute the fitness value
	individual.fitness = ...
	return individual
```
2. Define parameters for the genome, the population and the evolutionary algorithm
```python
params = {
     'seed': 8188212,
     'max_generations': 1000,
     'min_fitness': 0.,

     'population_params': {
          'n_parents': 10,
          'mutation_rate': 0.5,
      },

     'ea_params': {
         'n_offsprings': 10,
         'n_breeding': 10,
         'tournament_size': 1,
         'n_processes': 2},

     'genome_params': {
         'n_inputs': 2,
         'n_outputs': 1,
         'n_columns': 10,
         'n_rows': 5,
         'levels_back': 2,
         'primitives': [gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPDiv, gp.CGPConstantFloat]},
      }
```
3. Initialize the population and an evolutionary algorithm instance:
```python
pop = gp.CGPPopulation(**params['population_params'],
                          seed=params['seed'], genome_params=params['genome_params'])
ea = gp.ea.MuPlusLambda(**params['ea_params'])
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
history = gp.evolve(pop, obj, ea, params['max_generations'],
                       params['min_fitness'], print_progress=True, callback=recording_callback)
```



Code status
===========

[![Build Status](https://travis-ci.org/jakobj/python-gp.svg?branch=master)](https://travis-ci.org/jakobj/python-gp)
