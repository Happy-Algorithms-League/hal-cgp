class CGPNode():
    _arity = None
    _inputs = None
    _output = None
    _name = None
    _idx = None

    def __init__(self, idx, inputs):
        self._idx = idx
        self._inputs = inputs

    @property
    def arity(self):
        return self._arity

    @property
    def inputs(self):
        return self._inputs

    @property
    def idx(self):
        return self._idx

    def __repr__(self):
        return '{}(idx: {}, arity: {}, inputs {})'.format(self._name, self._idx, self._arity, self._inputs)

    @property
    def output(self):
        return self._output


class CGPAdd(CGPNode):
    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

    def __call__(self, x, graph):
        if self._inputs[0] < 0:
            inp1 = x[self._inputs[0] + graph._n_inputs]
        else:
            inp1 = graph[self._inputs[0]].output
        if self._inputs[1] < 0:
            inp2 = x[self._inputs[1] + graph._n_inputs]
        else:
            inp2 = graph[self._inputs[1]].output
        self._output = inp1 + inp2


class CGPSub(CGPNode):
    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

    def __call__(self, x, graph):
        if self._inputs[0] < 0:
            inp1 = x[self._inputs[0] + graph._n_inputs]
        else:
            inp1 = graph[self._inputs[0]].output
        if self._inputs[1] < 0:
            inp2 = x[self._inputs[1] + graph._n_inputs]
        else:
            inp2 = graph[self._inputs[1]].output
        self._output = inp1 - inp2


class CGPOutputNode(CGPNode):
    _arity = 1

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output
