import collections

from .cgp_node import CGPInputNode, CGPOutputNode


class CGPGraph():
    _n_outputs = None
    _n_inputs = None
    _n_columns = None
    _n_rows = None
    _nodes = None
    _gnome = None

    def __init__(self, genome, primitives):
        self._primitives = primitives
        self.parse_genome(genome)
        self._genome = genome

    def parse_genome(self, genome):
        assert(self._primitives == genome._primitives)
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
            self._nodes.append(self._primitives[region[0]](idx, region[1:]))
            idx += 1

        for region in genome.output_regions():
            self._nodes.append(CGPOutputNode(idx, region[1:]))
            idx += 1

    def _column_idx(self, idx):
        return idx // self._n_rows

    @property
    def input_nodes(self):
        return self._nodes[:self._n_inputs]

    @property
    def hidden_nodes(self):
        return self._nodes[self._n_inputs:-self._n_outputs]

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
            active_nodes[self._column_idx(node.idx)].add(node)
            node.activate()

            # need to process all inputs to this node next
            for i in node.inputs:
                if i is not self._genome._non_coding_allele and not isinstance(self._nodes[i], CGPInputNode):
                    nodes_to_process.append(self._nodes[i])

        # store values of x in input nodes
        for i, xi in enumerate(x):
            assert(isinstance(self._nodes[i], CGPInputNode))
            self._nodes[i]._output = xi

        # evaluate active nodes in order
        for i in sorted(active_nodes):
            for node in active_nodes[i]:
                node(x, self)

        return [node._output for node in self.output_nodes]

    def __getitem__(self, key):
        return self._nodes[key]
