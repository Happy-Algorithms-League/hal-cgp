HAL-CGP
=======
[![Python3.6](https://img.shields.io/badge/python-3.6-red.svg)](https://www.python.org/downloads/release/python-369/)
[![Python3.7](https://img.shields.io/badge/python-3.7-red.svg)](https://www.python.org/)
[![Python3.8](https://img.shields.io/badge/python-3.8-red.svg)](https://www.python.org/)
[![GPL license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-3.0.html)
[![Build Status](https://api.travis-ci.org/Happy-Algorithms-League/hal-cgp.svg?branch=master)](https://travis-ci.org/Happy-Algorithms-League/hal-cgp)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage Status](https://coveralls.io/repos/github/Happy-Algorithms-League/python-gp/badge.svg?branch=master)](https://coveralls.io/github/Happy-Algorithms-League/python-gp?branch=master)

Cartesian genetic programming (CGP) in pure Python.

This library implements Cartesian genetic programming (e.g, Miller and Thomson, 2000; Miller, 2011) for symbolic regression in pure Python, targeting applications with expensive fitness evaluations. It provides Python data structures to represent and evolve two-dimensional directed graphs (genotype) that are translated into computational graphs (phenotype) implementing mathematical expressions. The computational graphs can be compiled as a Python functions, SymPy expressions (Meurer et al., 2017) or PyTorch modules (Paszke et al., 2017). The library currently implements an evolutionary algorithm, specifically (mu + lambda) evolution strategies adapted from Deb et al. (2002), to evolve a population of symbolic expressions in order to optimize an objective function.
