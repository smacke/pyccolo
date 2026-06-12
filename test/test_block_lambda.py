# -*- coding: utf-8 -*-
import pyccolo as pyc
from pyccolo.examples.block_lambda import BlockLambdaTracer


def test_generator_block():
    with BlockLambdaTracer:
        result = pyc.eval(
            """list(map{
                for i in range(10):
                    if i % 2 == 0:
                        yield i * i
            })"""
        )
    assert result == [0, 4, 16, 36, 64], result


def test_do_block_returns_last_expression():
    with BlockLambdaTracer:
        result = pyc.eval(
            """do{
                total = 0
                for i in range(5):
                    total += i
                total
            }"""
        )
    assert result == 10, result


def test_fn_block_returns_callable():
    with BlockLambdaTracer:
        f = pyc.eval(
            """fn{
                xs = []
                for i in range(3):
                    xs.append(i)
                xs
            }"""
        )
    assert callable(f)
    assert f() == [0, 1, 2]


def test_do_block_closes_over_locals():
    with BlockLambdaTracer:
        n = 7  # noqa: F841  (closed over by the do{...} block below)
        result = pyc.eval(
            """do{
                acc = 0
                for i in range(n):
                    acc += i
                acc
            }"""
        )
    assert result == sum(range(7)), result


def test_nested_blocks():
    with BlockLambdaTracer:
        result = pyc.eval(
            """do{
                evens = list(gen{
                    for i in range(6):
                        if i % 2 == 0:
                            yield i
                })
                evens
            }"""
        )
    assert result == [0, 2, 4], result


def test_normal_braces_untouched():
    with BlockLambdaTracer:
        assert pyc.eval("{1: 2, 3: 4}") == {1: 2, 3: 4}
        assert pyc.eval("{1, 2, 3}") == {1, 2, 3}
        # a normal map(...) call must still work
        assert list(pyc.eval("map(lambda x: x + 1, range(3))")) == [1, 2, 3]
