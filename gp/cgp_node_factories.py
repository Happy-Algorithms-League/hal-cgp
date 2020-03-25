from .cgp_node import CGPConstantFloat


def CGPConstantFloatFactory(output_value):
    """Create a custom CGPConstantFloat class with given output value.

    Warning: relies on closures, hence does neither work with pickle
    nor with multiprocessing. Define your own top-level classes if you
    want to use those libraries.

    Parameters
    ----------
    output_value : int/float
       Output value of this node.

    Returns
    -------
    CGPConstantFloat

    """

    class CustomCGPConstantFloat(CGPConstantFloat):
        def __init__(self, idx, inputs):
            super().__init__(idx, inputs)
            self._output = output_value

    return CustomCGPConstantFloat
