import collections
import sympy
from sympy.core.evaluate import evaluate
import torch

from .cgp_node import CGPInputNode, CGPOutputNode


class CGPGraph():

    def __init__(self, genome):
        self._n_outputs = None
        self._n_inputs = None
        self._n_columns = None
        self._n_rows = None
        self._nodes = None
        self._gnome = None

        self.parse_genome(genome)
        self._genome = genome

    def __repr__(self):
        return 'CGPGraph(' + str(self._nodes) + ')'

    def print_active_nodes(self):
        return 'CGPGraph(' + str([node for node in self._nodes if node._active]) + ')'

    def pretty_print(self):

        n_characters = 18

        def pretty_node_str(node):
            s = node.pretty_str(n_characters)
            assert len(s) == n_characters
            return s

        def empty_node_str():
            return ' ' * n_characters

        s = '\n'

        for row in range(self._n_rows):
            for column in range(-1, self._n_columns + 1):

                if column == -1:
                    if row < self._n_inputs:
                        s += pretty_node_str(self.input_nodes[row])
                    else:
                        s += empty_node_str()
                    s += '\t'

                elif column < self._n_columns:
                    s += pretty_node_str(self.hidden_nodes[row + column * self._n_rows])
                    s += '\t'

                else:
                    if row < self._n_outputs:
                        s += pretty_node_str(self.output_nodes[row])
                    else:
                        s += empty_node_str()
                    s += '\t'

            s += '\n'

        return s

    def parse_genome(self, genome):
        if genome.dna is None:
            raise RuntimeError('dna not initialized')

        self._genome = genome

        self._n_inputs = genome._n_inputs
        self._n_outputs = genome._n_outputs
        self._n_columns = genome._n_columns
        self._n_rows = genome._n_rows

        self._nodes = []

        idx = 0
        for region_idx, input_region in genome.iter_input_regions():
            self._nodes.append(CGPInputNode(idx, input_region[1:]))
            idx += 1

        for region_idx, hidden_region in genome.iter_hidden_regions():
            self._nodes.append(genome.primitives[hidden_region[0]](idx, hidden_region[1:]))
            idx += 1

        for region_idx, output_region in genome.iter_output_regions():
            self._nodes.append(CGPOutputNode(idx, output_region[1:]))
            idx += 1

        self._determine_active_nodes()

    def _hidden_column_idx(self, idx):
        return (idx - self._n_inputs) // self._n_rows

    @property
    def input_nodes(self):
        return self._nodes[:self._n_inputs]

    @property
    def hidden_nodes(self):
        return self._nodes[self._n_inputs:-self._n_outputs]

    @property
    def output_nodes(self):
        return self._nodes[-self._n_outputs:]

    def _determine_active_nodes(self):

        # determine active nodes
        active_nodes_by_hidden_column_idx = collections.defaultdict(set)  # use set to avoid duplication
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
                    if not isinstance(self._nodes[i], CGPInputNode):
                        nodes_to_process.append(self._nodes[i])
                    else:
                        self._nodes[i].activate()

        return active_nodes_by_hidden_column_idx

    def determine_active_regions(self):
        active_regions = []
        active_nodes_by_hidden_column_idx = self._determine_active_nodes()
        for column_idx in active_nodes_by_hidden_column_idx:
            for node in active_nodes_by_hidden_column_idx[column_idx]:
                active_regions.append(node.idx)

        return active_regions

    def __call__(self, x):

        # store values of x in input nodes
        for i, xi in enumerate(x):
            assert(isinstance(self._nodes[i], CGPInputNode))
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

        return '[{}]'.format(', '.join(node.output_str for node in self.output_nodes))

    def _format_output_str_of_all_nodes(self):

        for i, node in enumerate(self.input_nodes):
            node.format_output_str(self)

        active_nodes = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes):
            for node in active_nodes[hidden_column_idx]:
                node.format_output_str(self)

    def compile_func(self):

        self._format_output_str_of_all_nodes()

        func_str = 'def _f(x): return [{}]'.format(', '.join(node.output_str for node in self.output_nodes))
        exec(func_str)
        return locals()['_f']

    def to_func(self):
        return self.compile_func()

    def compile_torch_class(self):

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

        class_str = \
"""
class _C(torch.nn.Module):

    def __init__(self):
        super().__init__()
"""
        for s in all_parameter_str:
            class_str += '        ' + s

        func_str = \
"""
    def forward(self, x):
        return torch.stack([{}], dim=1)
"""

        func_str = func_str.format(', '.join(node.output_str_torch for node in self.output_nodes))

        class_str += func_str

        exec(class_str)
        exec('_c = _C()')

        return locals()['_c']

    def to_torch(self):
        return self.compile_torch_class()

    def update_parameters_from_torch_class(self, torch_cls):
        for n in self._nodes:
            if n.is_parameter:
                try:
                    n._output = eval('torch_cls._p{}[0]'.format(n._idx))
                except AttributeError:
                    pass

    def _format_sympy_expressions_of_active_nodes(self):

        for node in self.input_nodes:
            node.format_sympy_expression(self)

        active_nodes = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes):
            for node in active_nodes[hidden_column_idx]:
                node.format_sympy_expression(self)

    def compile_sympy_expression(self):

        self._format_sympy_expressions_of_active_nodes()

        sympy_expr = ''
        # to get an expression that reflects the computational graph,
        # sympy should not automatically simplify the expression; if a
        # simplified expression is, run {expr}.simplify() on the output
        sympy_expr += 'with evaluate(False):\n'

        # register sympy symbol for inputs
        for node in self.input_nodes:
            sympy_expr += "    {name} = sympy.symbols('{name}')\n".format(name=node.sympy_var_name)

        # collect variable names for outputs
        sympy_output_var_names = []
        for node in self.output_nodes:
            sympy_output_var_names.append(node.sympy_var_name)

        # construct and execute sympy expression
        sympy_expr += '    {}'.format('\n'.join(node.sympy_expr for node in self.output_nodes))
        exec(sympy_expr)

        # can not use the dict returned by locals() in list
        # comprehension due to scope; we hence first store the locals
        # of the function and use that in the list comprehension
        local_dict = locals()
        return [local_dict[name] for name in sympy_output_var_names]

    def to_sympy(self):
        return self.compile_sympy_expression()
