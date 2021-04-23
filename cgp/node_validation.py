from typing import TYPE_CHECKING, Type

import numpy as np

try:
    from sympy.core import expr as sympy_expr  # noqa: F401

    sympy_available = True
except ModuleNotFoundError:
    sympy_available = False

try:
    import torch  # noqa: F401

    torch_available = True
except ModuleNotFoundError:
    torch_available = False


if TYPE_CHECKING:
    from .genome import Genome  # noqa: F401
    from .node import OperatorNode  # noqa: F401


def _create_genome(cls: Type["OperatorNode"]) -> "Genome":
    # delayed imports to avoid circular imports
    from .genome import ID_INPUT_NODE, ID_NON_CODING_GENE, ID_OUTPUT_NODE, Genome

    primitives = (cls,)
    genome = Genome(1, 1, 1, 1, primitives)
    dna = [ID_INPUT_NODE]
    arity = max(cls._arity, 1)
    for _ in range(arity):
        dna += [ID_NON_CODING_GENE]
    dna += [0]
    for _ in range(arity):
        dna += [0]
    dna += [ID_OUTPUT_NODE, 1]
    for _ in range(arity - 1):
        dna += [ID_NON_CODING_GENE]
    genome.dna = dna

    return genome


def check_to_func(cls: Type["OperatorNode"]) -> None:
    # delayed imports to avoid circular imports
    from .cartesian_graph import CartesianGraph

    genome = _create_genome(cls)

    f = CartesianGraph(genome).to_func()
    x = 1.0
    f(x)


def check_to_numpy(cls: Type["OperatorNode"]) -> None:
    # delayed imports to avoid circular imports
    from .cartesian_graph import CartesianGraph

    genome = _create_genome(cls)

    f = CartesianGraph(genome).to_numpy()
    x = np.ones(3)
    f(x)


def check_to_torch(cls: Type["OperatorNode"]) -> None:

    if not torch_available:
        return

    # delayed imports to avoid circular imports
    from .cartesian_graph import CartesianGraph

    genome = _create_genome(cls)

    f = CartesianGraph(genome).to_torch()
    x = torch.ones((3, 1))
    f(x)[0]


def check_to_sympy(cls: Type["OperatorNode"]) -> None:

    if not sympy_available:
        return

    # delayed imports to avoid circular imports
    from .cartesian_graph import CartesianGraph

    genome = _create_genome(cls)

    f = CartesianGraph(genome).to_sympy()
    assert isinstance(f, sympy_expr.Expr)
    x = [1.0]
    f.subs("x_0", x[0]).evalf()
