import dataclasses
import pytest

import cgp
from cgp.primitives import Primitives


def test_init_with_list_raises():
    with pytest.raises(TypeError):
        Primitives([cgp.Add])


def test_init_with_instance_raises():
    with pytest.raises(TypeError):
        Primitives((cgp.Add(0, []),))


def test_init_with_wrong_class_raises():
    with pytest.raises(TypeError):
        Primitives((list,))


def test_immutable_primitives():
    primitives = Primitives((cgp.Add, cgp.Sub))
    with pytest.raises(TypeError):
        primitives[0] = cgp.Add

    with pytest.raises(TypeError):
        primitives._primitives[0] = cgp.Add

    with pytest.raises(dataclasses.FrozenInstanceError):
        primitives.primitives = (cgp.Add,)


def test_max_arity():
    plain_primitives = (cgp.Add, cgp.Sub, cgp.ConstantFloat)
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
    plain_primitives = (cgp.Add, cgp.Sub, cgp.Mul, cgp.Div)
    primitives = Primitives(plain_primitives)

    for k in range(10):
        primitives_clone = Primitives(tuple(primitives))
        for i in range(len(plain_primitives)):
            assert primitives[i] == primitives_clone[i]
