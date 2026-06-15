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


def _coactive_regression_callee():
    return 42


def test_instrument_for_self_while_other_tracer_active():
    # Regression: instrumenting a function for one tracer while a second tracer
    # is co-active must still weave the instrumenting tracer's events. The
    # instrumenting tracer transiently hard-disables itself during the recompile;
    # it must not be dropped from the rewrite. Here A wants before_call but the
    # co-active B does not, so if A were dropped the call site would get no
    # before_call emit at all and A's interception would silently never fire.
    seen = []

    class WatchesCalls(pyc.BaseTracer):
        @pyc.register_handler(pyc.before_call)
        def handle_call(self, ret, *_, **__):
            seen.append(getattr(ret, "__name__", ret))
            return ret

    class WatchesAssigns(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret

    watch_calls = WatchesCalls.instance()
    watch_assigns = WatchesAssigns.instance()

    with watch_assigns:  # a second tracer is live while we instrument for watch_calls

        @watch_calls
        def f():
            return _coactive_regression_callee()

    assert f() == 42
    assert seen == ["_coactive_regression_callee"]  # before_call fired anyway
