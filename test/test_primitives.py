import pytest

import gp
from gp.primitives import Primitives


def test_immutable_primitives():
    primitives = Primitives([gp.Add, gp.Sub])
    with pytest.raises(TypeError):
        primitives[0] = gp.Add

    # currently setting this possible, since MappingProxy which was
    # used to enforce this behaviour can not be pickled and hence was
    # removed from Primitives
    # with pytest.raises(TypeError):
    #     primitives._primitives[0] = gp.Add


def test_max_arity():
    plain_primitives = [gp.Add, gp.Sub, gp.ConstantFloat]
    primitives = Primitives(plain_primitives)

    arity = 0
    for p in plain_primitives:
        if arity < p._arity:
            arity = p._arity

    assert arity == primitives.max_arity


def test_check_for_correct_class():
    with pytest.raises(TypeError):
        Primitives(["test"])

    with pytest.raises(TypeError):
        Primitives([str])
