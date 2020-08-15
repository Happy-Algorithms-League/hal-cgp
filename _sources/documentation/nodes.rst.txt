=====
Nodes
=====

Nodes define the buildings blocks of the mathematical expressions represented by an individual.
They are represented in the invididual's genome (see :ref:`pop_ind`).

A node receives data from a certain number of input nodes (specified by its `arity` which can be any non-negative number) and transforms them into one output value. A priori, there are no limitations regarding the type of transformation, as long as it can be expressed in pure Python, SymPy_, PyTorch_, and NumPy_.

----------------------------
Default nodes in hal-cgp
----------------------------

hal-cgp comes with a set of implemented nodes:

- The `InputNode` and `OutputNode` are special nodes that represent the input to and output from the computational graph.
- A set of basic mathematical operations is implemented as `OperatorNodes`: addition (``Add``), subtraction (``Sub``), multiplication (``Mul``), division (``Div``) and power (``Pow``).
- A `ConstantFloat` that represents a constant in mathematical expressions.
- A `ParameterNode` that represents an adjustable parameter (see :ref:`local_search`).

------------
Custom nodes
------------

In hal-cgp, it is straightforward to implement new, custom nodes.
The custom nodes needs to be implemented as a subclass of the `OperatorNode` (:meth:`cgp.node.OperatorNode`) and define its ``arity`` as well as its transformation, defined by the ``_def_output`` member of the class. The ``_def_output`` string defines the transformation in pure Python and is used to automatically define the transformation in SymPy_, PyTorch_, and Numpy_. In this string `x_i` refers to the `i`th input to the node.


As a simple example, we implement a custom node that doubles its input:

  .. code-block:: python

		  
     import cgp
     
     class Double(cgp.OperatorNode): 
         """ A node that doubles its input. 
	 """ 
	 _arity = 1 
	 _def_output = "2*x_0"

If the representation of the transformation in SymPy_, PyTorch_, or NumPy_ differs from pure Python, we need to define dedicated expressions via the `_def_X_output` members. For instance, to implement a node that computes the exponential of an input, we define the same operation in four different expressions:
	 
  .. code-block:: python

		  
     import cgp

     class Exp(cgp.OperatorNode): 
         """ A node that calculates the exponential of its input. 
	 """ 
	 _arity = 1 
	 _def_output = "math.exp(x_0)" 
	 _def_numpy_output = "np.exp(x_0)" 
	 _def_torch_output = "torch.exp(x_0)"
	 _def_sympy_output = "exp(x_0)"

	

We can make this custom mode more flexible by adding an adjustable parameter for the scale of the exponential. Parameter names are enclosed by angle brackets in the expressions. Furthermore, each parameter requires a function that returns its initial values to be defined in the `_initial_values` dict.
We add the ``<scale>`` term in the expressions and define its initial value in the ``_initial_values`` class member:
	 
  .. code-block:: python

		  
     import cgp

     class ExpScaled(cgp.OperatorNode): 
         """ A node that calculates the exponential of its input. An adjustable
	 parameter governs the scale.
	 """
	 _arity = 1
	 _initial_values = {"<scale>": lambda: 1.0}
	 _def_output = "math.exp(<scale> * x_0)" 
	 _def_numpy_output = "np.exp(<scale> * x_0)" 
	 _def_torch_output = "torch.exp(<scale> * x_0)"
	 _def_sympy_output = "exp(<scale> * x_0)"

	
For a complete example that uses custom nodes with adjustable parameters, please refer to :ref:`sphx_glr_auto_examples_example_parametrized_nodes.py`.


.. _numpy: https://numpy.org
.. _pytorch: https://pytorch.org
.. _SymPy: https://www.sympy.org
