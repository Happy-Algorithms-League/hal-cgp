import collections

try:
    import sympy
except ModuleNotFoundError:
    sympy = None

try:
    import torch  # noqa: F401
except ModuleNotFoundError:
    torch = None


from .node import InputNode, OutputNode


class CartesianGraph:
    """Class representing a particular Cartesian graph defined by a
    Genome.
    """

    def __init__(self, genome):
        """Init function.

        Parameters
        ----------
        genome: Genome
            Genome defining graph connectivity and node operations.
        """
        self._n_outputs = None
        self._n_inputs = None
        self._n_columns = None
        self._n_rows = None
        self._nodes = None
        self._gnome = None

        self.parse_genome(genome)
        self._genome = genome

    def __repr__(self):
        return "CartesianGraph(" + str(self._nodes) + ")"

    def print_active_nodes(self):
        """Print a representation of all active nodes in the graph.
        """
        return "CartesianGraph(" + str([node for node in self._nodes if node._active]) + ")"

    def pretty_print(self):
        """Print a pretty representation of the Cartesian graph.
        """
        n_characters = 18

        def pretty_node_str(node):
            s = node.pretty_str(n_characters)
            assert len(s) == n_characters
            return s

        def empty_node_str():
            return " " * n_characters

        s = "\n"

        for row in range(self._n_rows):
            for column in range(-1, self._n_columns + 1):

                if column == -1:
                    if row < self._n_inputs:
                        s += pretty_node_str(self.input_nodes[row])
                    else:
                        s += empty_node_str()
                    s += "\t"

                elif column < self._n_columns:
                    s += pretty_node_str(self.hidden_nodes[row + column * self._n_rows])
                    s += "\t"

                else:
                    if row < self._n_outputs:
                        s += pretty_node_str(self.output_nodes[row])
                    else:
                        s += empty_node_str()
                    s += "\t"

            s += "\n"

        return s

    def parse_genome(self, genome):
        if genome.dna is None:
            raise RuntimeError("dna not initialized")

        self._genome = genome

        self._n_inputs = genome._n_inputs
        self._n_outputs = genome._n_outputs
        self._n_columns = genome._n_columns
        self._n_rows = genome._n_rows

        self._nodes = []

        idx = 0
        for region_idx, input_region in genome.iter_input_regions():
            self._nodes.append(InputNode(idx, input_region[1:]))
            idx += 1

        for region_idx, hidden_region in genome.iter_hidden_regions():
            self._nodes.append(genome.primitives[hidden_region[0]](idx, hidden_region[1:]))
            idx += 1

        for region_idx, output_region in genome.iter_output_regions():
            self._nodes.append(OutputNode(idx, output_region[1:]))
            idx += 1

        self._determine_active_nodes()

    def _hidden_column_idx(self, idx):
        return (idx - self._n_inputs) // self._n_rows

    @property
    def input_nodes(self):
        return self._nodes[: self._n_inputs]

    @property
    def hidden_nodes(self):
        return self._nodes[self._n_inputs : -self._n_outputs]

    @property
    def output_nodes(self):
        return self._nodes[-self._n_outputs :]

    def _determine_active_nodes(self):

        # determine active nodes
        active_nodes_by_hidden_column_idx = collections.defaultdict(
            set
        )  # use set to avoid duplication
        nodes_to_process = self.output_nodes  # output nodes always need to be processed

        while len(nodes_to_process) > 0:
            node = nodes_to_process.pop()

            # add this node to active nodes; sorted by column to
            # determine evaluation order
            active_nodes_by_hidden_column_idx[self._hidden_column_idx(node.idx)].add(node)
            node.activate()

            # need to process all inputs to this node next
            for i in node.inputs:
                if i is not self._genome._non_coding_allele:
                    if not isinstance(self._nodes[i], InputNode):
                        nodes_to_process.append(self._nodes[i])
                    else:
                        self._nodes[i].activate()

        return active_nodes_by_hidden_column_idx

    def determine_active_regions(self):
        """Determine the active regions in the computational graph.

        Returns
        -------
        List[int]
            List of ids of the active nodes.
        """
        active_regions = []
        active_nodes_by_hidden_column_idx = self._determine_active_nodes()
        for column_idx in active_nodes_by_hidden_column_idx:
            for node in active_nodes_by_hidden_column_idx[column_idx]:
                active_regions.append(node.idx)

        return active_regions

    def __call__(self, x):
        # store values of x in input nodes
        for i, xi in enumerate(x):
            assert isinstance(self._nodes[i], InputNode)
            self._nodes[i]._output = xi

        # evaluate active nodes in order
        active_nodes_by_hidden_column_idx = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes_by_hidden_column_idx):
            for node in active_nodes_by_hidden_column_idx[hidden_column_idx]:
                node(x, self)

        return [node._output for node in self.output_nodes]

    def __getitem__(self, key):
        return self._nodes[key]

    def to_str(self):

        self._format_output_str_of_all_nodes()
        out_str = ", ".join(node.output_str for node in self.output_nodes)
        return f"[{out_str}]"

    def _format_output_str_of_all_nodes(self):

        for i, node in enumerate(self.input_nodes):
            node.format_output_str(self)

        active_nodes = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes):
            for node in active_nodes[hidden_column_idx]:
                node.format_output_str(self)

    def to_func(self):
        """Compile the function represented by the computational graph.

        Generates a definition of the function in Python code and
        executes the function definition to create a Callable.

        Returns
        -------
        Callable
            Callable executing the function represented by the computational graph.
        """
        self._format_output_str_of_all_nodes()
        s = ", ".join(node.output_str for node in self.output_nodes)
        func_str = f"""\
def _f(x):
    if len(x) != {self._n_inputs}:
        raise ValueError(f'input has length {{len(x)}}, expected {self._n_inputs}')
    return [{s}]
"""
        exec(func_str)
        return locals()["_f"]

    def to_torch(self):
        """Compile the function represented by the computational graph to a Torch class.

        Generates a definition of the Torch class in Python code and
        executes it to create an instance of the class.

        Returns
        -------
        torch.nn.Module
            Instance of the PyTorch class.
        """
        if torch is None:
            raise ModuleNotFoundError("No module named 'torch' (extra requirement)")

        for i, node in enumerate(self.input_nodes):
            node.format_output_str_torch(self)

        active_nodes_by_hidden_column_idx = self._determine_active_nodes()
        all_parameter_str = []
        for hidden_column_idx in sorted(active_nodes_by_hidden_column_idx):
            for node in active_nodes_by_hidden_column_idx[hidden_column_idx]:
                node.format_output_str_torch(self)
                if node.is_parameter:
                    node.format_parameter_str()
                    all_parameter_str.append(node.parameter_str)
        forward_str = ", ".join(node.output_str_torch for node in self.output_nodes)
        class_str = f"""\
class _C(torch.nn.Module):

    def __init__(self):
        super().__init__()

"""
        for s in all_parameter_str:
            class_str += "        " + s

        func_str = f"""\

    def forward(self, x):
        if len(x.shape) != 2:
            raise ValueError(
                f"input has shape {{tuple(x.shape)}}, expected (<batch_size>, {self._n_inputs})"
            )
        if x.shape[1] != {self._n_inputs}:
            raise ValueError(
                f"input has shape {{tuple(x.shape)}}, expected ({{x.shape[0]}}, {self._n_inputs})"
            )
        return torch.stack([{forward_str}], dim=1)
        """
        class_str += func_str

        exec(class_str)
        exec("_c = _C()")

        return locals()["_c"]

    def update_parameters_from_torch_class(self, torch_cls):
        """Update values stored in constant nodes of graph from parameters of a given Torch instance.

        Can be used to import new values from a Torch class after a autograd step.

        Parameters
        ----------
        torch_cls : torch.nn.module
            Instance of a torch class.

        Returns
        -------
        None
        """
        for n in self._nodes:
            if n.is_parameter:
                try:
                    n._output = eval(f"torch_cls._p{n._idx}[0]")
                except AttributeError:
                    pass

    def to_sympy(self, simplify=True):
        """Compile computational graph into a list of sympy-compatible string expressions.

        Generates one sympy expression for each output node.

        Parameters
        ----------
        simplify : boolean, optional
            Whether to simplify the expression using sympy's
            simplify() method. Defaults to True.

        Returns
        ----------
        List[sympy.core.Expr]
            List of sympy expressions.
        """
        if sympy is None:
            raise ModuleNotFoundError("No module named 'sympy' (extra requirement)")

        def _validate_sympy_expr(expr):
            """Helper function that raises an exception upon encountering a SymPy
            expression that can not be evaluated.

            """

            class InvalidSympyExpression(Exception):
                pass

            if "zoo" in str(expr) or "nan" in str(expr):
                raise InvalidSympyExpression(str(expr))

            return expr

        self._format_output_str_of_all_nodes()

        sympy_exprs = []
        for output_node in self.output_nodes:

            # replace all input-variable strings with sympy-compatible symbol
            # strings (i.e., x[0] -> x_0)
            s = output_node.output_str
            for input_node in self.input_nodes:
                s = s.replace(
                    input_node.output_str, input_node.output_str.replace("[", "_").replace("]", "")
                )

            # to get an expression that reflects the computational graph,
            # sympy should not automatically simplify the expression
            with sympy.evaluate(False):
                sympy_exprs.append(sympy.sympify(s))

        if not simplify:
            return sympy_exprs
        else:  # simplify expression if desired
            for i, expr in enumerate(sympy_exprs):
                sympy_exprs[i] = _validate_sympy_expr(expr.simplify())
            return sympy_exprs
