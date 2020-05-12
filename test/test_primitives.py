import dataclasses
import pytest

import gp
from gp.primitives import Primitives


def test_init_with_list_raises():
    with pytest.raises(TypeError):
        Primitives([gp.Add])


def test_init_with_instance_raises():
    with pytest.raises(TypeError):
        Primitives((gp.Add(0, []),))


def test_init_with_wrong_class_raises():
    with pytest.raises(TypeError):
        Primitives((list,))


def test_immutable_primitives():
    primitives = Primitives((gp.Add, gp.Sub))
    with pytest.raises(TypeError):
        primitives[0] = gp.Add

    with pytest.raises(TypeError):
        primitives._primitives[0] = gp.Add

    with pytest.raises(dataclasses.FrozenInstanceError):
        primitives.primitives = (gp.Add,)


def test_max_arity():
    plain_primitives = (gp.Add, gp.Sub, gp.ConstantFloat)
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


def test_function_indices_remain_fixed_in_list_conversion():
    plain_primitives = (gp.Add, gp.Sub, gp.Mul, gp.Div)
    primitives = Primitives(plain_primitives)

    for k in range(10):
        primitives_clone = Primitives(tuple(primitives))
        for i in range(len(plain_primitives)):
            assert primitives[i] == primitives_clone[i]
