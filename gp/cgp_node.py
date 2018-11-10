class CGPNode():
    _arity = None
    _active = False
    _inputs = None
    _output = None
    _name = None
    _idx = None
    _is_parameter = False

    def __init__(self, idx, inputs):
        self._idx = idx
        self._inputs = inputs

        assert(idx not in inputs)

    @property
    def arity(self):
        return self._arity

    @property
    def inputs(self):
        return self._inputs[:self._arity]

    @property
    def idx(self):
        return self._idx

    def __repr__(self):
        return '{}(idx: {}, active: {}, arity: {}, inputs {}, output {})'.format(self._name, self._idx, self._active, self._arity, self._inputs, self._output)

    @property
    def output(self):
        return self._output

    def activate(self):
        self._active = True

    def format_output_str(self, graph):
        raise NotImplementedError()

    def format_output_str_torch(self, graph):
        # in case output_str_torch implementation is not provided, use
        # standard output_str
        self.format_output_str(graph)

    def format_parameter_str(self):
        raise NotImplementedError

    @property
    def output_str(self):
        return self._output_str

    @property
    def output_str_torch(self):
        return self.output_str

    @property
    def is_parameter(self):
        return self._is_parameter

    @property
    def parameter_str(self):
        return self._parameter_str


class CGPAdd(CGPNode):
    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output + graph[self._inputs[1]].output

    def format_output_str(self, graph):
        self._output_str = '({} + {})'.format(graph[self._inputs[0]].output_str, graph[self._inputs[1]].output_str)


class CGPSub(CGPNode):
    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output - graph[self._inputs[1]].output

    def format_output_str(self, graph):
        self._output_str = '({} - {})'.format(graph[self._inputs[0]].output_str, graph[self._inputs[1]].output_str)


class CGPMul(CGPNode):
    _arity = 2

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output * graph[self._inputs[1]].output

    def format_output_str(self, graph):
        self._output_str = '({} * {})'.format(graph[self._inputs[0]].output_str, graph[self._inputs[1]].output_str)


class CGPConstantFloat(CGPNode):
    _arity = 0
    _is_parameter = True

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

        self._output = 1.

    def __call__(self, x, graph):
        pass

    def format_output_str(self, graph):
        self._output_str = '{}'.format(self._output)

    def format_output_str_torch(self, graph):
        self._output_str = 'self._p{}'.format(self._idx)

    def format_parameter_str(self):
        self._parameter_str = 'self._p{} = torch.nn.Parameter(torch.Tensor([{}]))\n'.format(self._idx, self._output)


class CGPInputNode(CGPNode):
    _arity = 0

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

    def __call__(self, x, graph):
        assert(False)

    def format_output_str(self, graph):
        self._output_str = 'x[{}]'.format(self._idx)

    def format_output_str_torch(self, graph):
        self._output_str = 'x[:, {}]'.format(self._idx)


class CGPOutputNode(CGPNode):
    _arity = 1

    def __init__(self, idx, inputs):
        super().__init__(idx, inputs)

        self._name = self.__class__.__name__

    def __call__(self, x, graph):
        self._output = graph[self._inputs[0]].output

    def format_output_str(self, graph):
        self._output_str = '{}'.format(graph[self._inputs[0]].output_str)
