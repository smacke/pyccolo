# -*- coding: utf-8 -*-
import ast
import pyccolo as pyc


def test_basic_decorator():
    class IncrementsAssignValue(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    tracer = IncrementsAssignValue.instance()

    @tracer
    def f():
        x = 41
        return x

    assert f() == 42


def test_decorated_tracing_decorator():
    class IncrementsAssignValue(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    tracer = IncrementsAssignValue.instance()

    def twice(f):
        def new_f():
            return f() * 2

        return new_f

    @twice
    @tracer
    def f():
        x = 41
        return x

    assert f() == 84


def test_multiple_tracing_decorators():
    class IncrementsAssignValue1(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    class IncrementsAssignValue2(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 2

    tracer1 = IncrementsAssignValue1.instance()
    tracer2 = IncrementsAssignValue2.instance()

    @pyc.instrumented((tracer1, tracer2))
    def f():
        x = 41
        return x

    assert f() == 44
