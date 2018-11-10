import collections
import torch

from .cgp_node import CGPInputNode, CGPOutputNode


class CGPGraph():
    _n_outputs = None
    _n_inputs = None
    _n_columns = None
    _n_rows = None
    _nodes = None
    _gnome = None

    def __init__(self, genome):
        self.parse_genome(genome)
        self._genome = genome

    def __repr__(self):
        return 'CGPGraph(' + str(self._nodes) + ')'

    def print_active_nodes(self):
        return 'CGPGraph(' + str([node for node in self._nodes if node._active]) + ')'

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
        for region in genome.input_regions():
            self._nodes.append(CGPInputNode(idx, region[1:]))
            idx += 1

        for region in genome.hidden_regions():
            self._nodes.append(genome.primitives[region[0]](idx, region[1:]))
            idx += 1

        for region in genome.output_regions():
            self._nodes.append(CGPOutputNode(idx, region[1:]))
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

    def compile_func(self):

        for i, node in enumerate(self.input_nodes):
            node.format_output_str(self)

        active_nodes = self._determine_active_nodes()
        for hidden_column_idx in sorted(active_nodes):
            for node in active_nodes[hidden_column_idx]:
                node.format_output_str(self)

        func_str = 'def _f(x): return [{}]'.format(', '.join(node.output_str for node in self.output_nodes))
        exec(func_str)
        return locals()['_f']

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

    def update_parameter_values(self, torch_cls):
        for n in self._nodes:
            if n.is_parameter:
                try:
                    n._output = eval('torch_cls._p{}[0]'.format(n._idx))
                except AttributeError:
                    pass
