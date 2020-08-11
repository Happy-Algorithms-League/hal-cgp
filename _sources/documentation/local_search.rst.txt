==========================================================
Searching numerical values for parameters via local search
==========================================================

hal-cgp supports the use of numerical values in expression via nodes with input-independent output in order to represent expressions like `f(x) = x + c` with some fixed value `c`.

While the `ConstantFloat` node has a fixed output value, the output value of a `Parameter` node is stored per individual and can be modified during evolution via local search. These parameters can be different for each position in the genome. One can define custom nodes which contain an arbitrary number of parameters (see :ref:`sphx_glr_auto_examples_example_parametrized_nodes.py`)

To find suitable values for these parameters we perform a numerical optimization for the parameters of the `k` best individuals of each generation. Since this procedure is acting on each individual separately it is referred to as "local search". After this optimization the fitness of each modified individual is reevaluated taking into account the adapted parameter values, and finally used for selection in the evolutionary algorithm.

.. image:: local_search.svg

The library provides a local search function (:meth:`cgp.local_search.gradient_based`) that performs stochastic gradient descent on the values of `Parameter` nodes (cf. Topchy & Punch, 2001; Izzo et al., 2017). Note that this method requires the definition of a differentiable loss function 

--------
Examples
--------

See :ref:`sphx_glr_auto_examples_example_differential_evo_regression.py` for an example evolving an expression containing adjustable parameters.

See :ref:`sphx_glr_auto_examples_example_parametrized_nodes.py` for an example evolving an expression containing operator nodes with adjustable parameters.


----------
References
----------

Izzo, D., Biscani, F., & Mereta, A. (2017). Differentiable Genetic Programming. In European Conference on Genetic Programming (pp. 35-51). Springer, Cham.

Topchy, A., & Punch, W. F. (2001). Faster Genetic Programming based on Local Gradient Search of Numeric Leaf Values. In Proceedings of the Genetic and Evolutionary Computation Conference (GECCO-2001) (Vol. 155162). Morgan Kaufmann San Francisco, CA, USA.
