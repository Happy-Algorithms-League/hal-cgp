import numpy as np
import pytest
import sys

sys.path.insert(0, "../")
import gp
from gp.cgp_primitives import CGPPrimitives


SEED = np.random.randint(2 ** 31)


def test_immutable_primitives():
    primitives = CGPPrimitives([gp.CGPAdd, gp.CGPSub])
    with pytest.raises(TypeError):
        primitives[0] = gp.CGPAdd

    # currently setting this possible, since MappingProxy which was
    # used to enforce this behaviour can not be pickled and hence was
    # removed from Primitives
    # with pytest.raises(TypeError):
    #     primitives._primitives[0] = gp.CGPAdd


def test_max_arity():
    plain_primitives = [gp.CGPAdd, gp.CGPSub, gp.CGPConstantFloat]
    primitives = CGPPrimitives(plain_primitives)

    arity = 0
    for p in plain_primitives:
        if arity < p._arity:
            arity = p._arity

    assert arity == primitives.max_arity


def test_check_for_correct_class():
    with pytest.raises(TypeError):
        CGPPrimitives(["test"])

    with pytest.raises(TypeError):
        CGPPrimitives([str])
