===========================
Populations and individuals
===========================

At the core of evolutionary search is the concept of a `population` that
consists of multiple `individuals` with associated fitness values. In genetic programming each of these individuals contains one or multiple `genomes` which are compiled to
computational graphs. The fitness of each individual is typically assessed by how good these computational graphs solve a given problem.

----------
Population
----------

A population (see :meth:`cgp.Population`) comprises a fixed number of parent
individuals which produce offspring in every
generation of the evolutionary algorithm.
At each generation, some parents, chosen by a particular selection technique, produce offspring by cloning
and randomly mutating the clones.

-----------
Individuals
-----------

There are two types of individuals:

- The `IndividualSingleGenome` (see
  :meth:`cgp.individual.IndividualSingleGenome`) possesses a single genome and
  thus represents a single computational graph

- The `IndividualMultiGenome` (see
  :meth:`cgp.individual.IndividualSingleGenome`) possesses multiple genomes and
  thus simultaneously represents multiple computational graphs.

These classes are thin wrappers around the `Genome` (see :meth:`cgp.Genome`) and
the `CartesianGraph` (see :meth:`cgp.CartesianGraph`) classes.

------
Genome
------

A `Genome` instance represents a particular genotype, i.e., a specific realization of a 2-dimensional Cartesian graph.
It provides methods for randomizing and mutating the genotype. It also stores numerical values of parameters in operator nodes that can be tuned via local search (see :ref:`local_search`).

---------------
Cartesian Graph
---------------

The `CartesianGraph` class compiles genotypes to phenotypes, i.e.,  computational graphs.
Several compilation targets are available:

- `to_func` creates a Python callable
- `to_nympy` creates a Python callable that accepts a NumPy_ array as input and returns a NumPy array
- `to_torch` creates a PyTorch_ module equipped with a `forward` method that accepts a PyTorch tensor as input and returns a PyTorch tensor
- `to_sympy` returns SymPy_ expressions representing the computational graph
  

.. _numpy: https://numpy.org
.. _pytorch: https://pytorch.org
.. _SymPy: https://www.sympy.org
