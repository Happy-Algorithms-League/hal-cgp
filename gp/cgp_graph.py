import collections

from .cgp_node import CGPOutputNode

# TODO: hunt + 1 bug somewhere
# TODO: fix key/idx issue for custom __getitem__


class CGPGraph():
    _n_outputs = None
    _n_inputs = None
    _n_columns = None
    _n_rows = None
    _nodes = None

    def __init__(self, n_inputs, n_outputs, n_columns, n_rows):

        self._n_inputs = n_inputs
        self._n_outputs = n_outputs
        self._n_columns = n_columns
        self._n_rows = n_rows

    def parse_genome(self, genome, primitives):

        self._nodes = []

        for i, region in enumerate(genome):
            self._nodes.append(primitives[region[0]](i, region[1:]))

        for i in range(self._n_outputs):
            self._nodes.append(CGPOutputNode(self._n_columns * self._n_rows + i, [genome[-self._n_outputs + i]]))

    def __call__(self, x):

        # determine active nodes
        active_nodes = collections.defaultdict(list)
        nodes_to_process = collections.deque(self._nodes[-self._n_outputs:])
        while len(nodes_to_process) > 0:
            node = nodes_to_process.pop()
            node.activate()
            active_nodes[node.idx // self._n_rows].append(node)
            for i in node.inputs:
                if i >= 0:  # do not add input nodes
                    nodes_to_process.append(self._nodes[i])

        # evaluate active nodes in order
        print(active_nodes)
        for i in sorted(active_nodes):
            for node in active_nodes[i]:
                node(x, self)

        return [node._output for node in self._nodes[-self._n_outputs:]]

    def __getitem__(self, idx):
        return self._nodes[idx]
