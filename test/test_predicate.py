# -*- coding: utf-8 -*-
import random
from pyccolo.predicate import CompositePredicate, Predicate


def _rand(threshold=0.5, **kwargs):
    return Predicate(lambda *_: random.random() < threshold, **kwargs)


def test_static_coalescing():
    assert CompositePredicate.any([Predicate.TRUE, _rand(), _rand()]) is Predicate.TRUE
    assert (
        CompositePredicate.any([Predicate.FALSE, _rand(), _rand()])
        is not Predicate.FALSE
    )
    assert (
        CompositePredicate.any([Predicate.FALSE, Predicate.FALSE, Predicate.FALSE])
        is Predicate.FALSE
    )
    assert CompositePredicate.any([]) is Predicate.TRUE
    assert (
        CompositePredicate.all([Predicate.FALSE, _rand(), _rand()]) is Predicate.FALSE
    )
    assert (
        CompositePredicate.all([Predicate.TRUE, _rand(), _rand()]) is not Predicate.TRUE
    )
    assert (
        CompositePredicate.any([Predicate.TRUE, Predicate.TRUE, Predicate.TRUE])
        is Predicate.TRUE
    )
    assert CompositePredicate.all([]) is Predicate.TRUE


def test_dynamic_behavior():
    assert Predicate.TRUE(None)
    assert not Predicate.FALSE(None)
    assert Predicate.TRUE.dynamic_call(None)
    assert not Predicate.FALSE.dynamic_call(None)

    static_false = CompositePredicate.any(
        [_rand(0, static=True), _rand(0, static=True), _rand(0, static=True)]
    )
    assert not static_false(None)
    assert static_false.dynamic_call(None)  # none of the filters kick in dynamically

    dynamic_false = CompositePredicate.any(
        [_rand(0, static=True), _rand(0, static=True), _rand(0)]
    )
    assert not dynamic_false(None)
    assert not dynamic_false.dynamic_call(None)
