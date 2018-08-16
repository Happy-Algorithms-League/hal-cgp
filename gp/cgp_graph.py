import collections

from .cgp_node import CGPOutputNode


class CGPGraph():
    _n_outputs = None
    _n_inputs = None
    _n_columns = None
    _n_rows = None
    _nodes = None

    def __init__(self, genome, primitives):
        self.parse_genome(genome, primitives)

    def parse_genome(self, genome, primitives):

        self._n_inputs = genome._n_inputs
        self._n_outputs = genome._n_outputs
        self._n_columns = genome._n_columns
        self._n_rows = genome._n_rows

        self._nodes = []

        for i, region in enumerate(genome):
            self._nodes.append(primitives[region[0]](i, region[1:]))

        for i in range(self._n_outputs):
            self._nodes.append(CGPOutputNode(self._n_columns * self._n_rows + i, [genome[-self._n_outputs + i]]))

    @property
    def output_nodes(self):
        return self._nodes[-self._n_outputs:]

    def __call__(self, x):

        # determine active nodes
        active_nodes = collections.defaultdict(set)  # use set to avoid duplication
        nodes_to_process = self.output_nodes  # output nodes always need to be processed

        while len(nodes_to_process) > 0:
            node = nodes_to_process.pop()

            # add this node to active nodes; sorted by column to
            # determine evaluation order
            current_column = node.idx // self._n_rows
            active_nodes[current_column].add(node)

            # need to process all inputs to this node next
            for i in node.inputs:
                if i >= 0:  # do not add (external) input nodes
                    nodes_to_process.append(self._nodes[i])

        # evaluate active nodes in order
        for i in sorted(active_nodes):
            for node in active_nodes[i]:
                node(x, self)

        return [node._output for node in self.output_nodes]

    def __getitem__(self, key):
        return self._nodes[key]
