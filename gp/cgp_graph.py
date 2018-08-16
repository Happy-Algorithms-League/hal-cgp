import collections

from .cgp_node import CGPInputNode, CGPOutputNode


class CGPGraph():
    _n_outputs = None
    _n_inputs = None
    _n_columns = None
    _n_rows = None
    _nodes = None

    def __init__(self, genome, primitives):
        self._primitives = primitives
        self.parse_genome(genome)

    def parse_genome(self, genome):

        self._n_inputs = genome._n_inputs
        self._n_outputs = genome._n_outputs
        self._n_columns = genome._n_columns
        self._n_rows = genome._n_rows

        self._nodes = []

        for i, region in enumerate(genome):
            self._nodes.append(self._primitives[region[0]](i, region[1:]))

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
            node.activate()

            # add this node to active nodes; sorted by column to
            # determine evaluation order
            current_column = node.idx // self._n_rows
            active_nodes[current_column].add(node)

            # need to process all inputs to this node next
            for i in node.inputs:
                if i >= 0:  # do not add (external) input nodes
                    nodes_to_process.append(self._nodes[i])

        # temporally add InputNodes; since they are indexed with
        # negative values, just append them to _nodes list and
        # list[-|idx|] will select the correct input node
        # TODO: dangerous if any node assumes standard shape of graph._nodes
        self._nodes += [CGPInputNode() for _ in range(self._n_inputs)]
        for i in range(-self._n_inputs, 0):
            self._nodes[i]._output = x[i + self._n_inputs]

        # evaluate active nodes in order
        for i in sorted(active_nodes):
            for node in active_nodes[i]:
                node(x, self)

        # remove input nodes again
        self._nodes = self._nodes[:-self._n_inputs]

        return [node._output for node in self.output_nodes]

    def __getitem__(self, key):
        return self._nodes[key]
