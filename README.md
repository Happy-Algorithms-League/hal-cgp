python-gp
===========
[![Python3.6](https://img.shields.io/badge/python-3.6-red.svg)](https://www.python.org/downloads/release/python-369/)
[![Python3.7](https://img.shields.io/badge/python-3.7-red.svg)](https://www.python.org/)
[![Python3.8](https://img.shields.io/badge/python-3.8-red.svg)](https://www.python.org/)
[![GPL license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-3.0.html)

Cartesian genetic programming (CGP) in Python.

This library implements Cartesian genetic programming in pure Python. It implements Python data structures to represent computational graphs in the Cartesian Genetic programming and provides one evolutionary algorithm,  (mu + lambda) evoluation strategies, to evolve a population of computational graphs given an objective function.

![CGP Sketch](cgp-sketch.png)

Basic usage
===========

A complete example of CGP applied to a regression problem can be found in `examples/example_evo_regression.py`.

The basic steps to be taken are:

1. **Define an objective function**. 

   The objective needs to take an individual as input variable and update the `fitness` member of the individual.
```
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
	fitness = ...
	individual.fitness = fitness
	return individual
```
2. Define parameters for the genome, the population and the evolutionary algorithm
```
params = {
   'seed': 8188212,
   'n_threads': 1,
   'max_generations': 1000,
   'min_fitness': 0.,

   'population_params': {
        'n_parents': 10,
        'mutation_rate': 0.5,
    },

   'ea_params': {
       'n_offsprings': 10,
       'n_breeding': 10,
       'tournament_size': 1},

   'genome_params': {
       'n_inputs': 2,
       'n_outputs': 1,
       'n_columns': 10,
       'n_rows': 5,
       'levels_back': 2,
       'primitives': [gp.CGPAdd, gp.CGPSub, gp.CGPMul, gp.CGPDiv, gp.CGPConstantFloat]},
    }
```
3. Initialize the `CGPPopulation` and the evolutionary algorithm instance:
```
pop = gp.CGPPopulation(**params['population_params'],
                          seed=params['seed'], genome_params=params['genome_params'])
ea = gp.ea.MuPlusLambda(**params['ea_params'])
```
4. Use the `evolve` function that ties everything together and executes the evolution:
```
history = gp.evolve(pop, obj, ea, params['max_generations'],
                       params['min_fitness'], record_history=record_history, print_progress=True)
```



Data structures
===============

The Cartesian genetic programming framework is implemented with the following data structures:

- `CGPPopulation` implements a population of `CGPIndividual` parent individuals and defines the standard procedures to create offspring from parent individuals, mutation and crossover.
- `CGPIndividual` represents an individual that is defined by its genome (`CGPGenome`).
- `CGPGenome` represents the genome of an individual. It is defined by its DNA (a list of integers) and parametrized by the parameters of CGP: `n_inputs`, `n_outputs`, `n_rows`, `levels_back`, and the primitives.
- `CGPGraph` implements the computational graph represented by a `CGPGenome` and provides a `parse_genome` method to parse the genome into a phenotype, the core algorithm defining the CGP framework. It provides three methods to parse the computational graph into representations that can be used in downstream tasks:
    - `to_sympy` parses the computational graph into a sympy-expression.
	- `to_torch` creates a torch.nn.Module child class.
	- `to_func` creates a Python callable.
- Primitives are operations that are combined in a computational graphs to produce an output value from given input values. Any operation is supported, the library currently implements:
   - Addition: `CGPAdd`
   - Subtraction: `CGPSub`
   - Multiplication: `CGPMul`
   - Division: `CGPDiv`
   - Exponentiation: `CGPPow`
   - Constants: `CGPConstantFloat`
   These operations are implemented as child classes of `CGPNode` and are grouped by the umbrella class `CGPPrimitives` inside the `CGPGenome`.



Code status
===========

[![Build Status](https://travis-ci.org/jakobj/python-gp.svg?branch=master)](https://travis-ci.org/jakobj/python-gp)
