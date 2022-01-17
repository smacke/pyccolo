# -*- coding: utf-8 -*-
import pyccolo as pyc


def test_basic_instrumented_import():
    class IncrementsAssignValue(pyc.BaseTracer):
        def should_instrument_file(self, filename: str) -> bool:
            return filename.endswith("foo.py")

        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    with IncrementsAssignValue.instance().tracing_enabled():
        import test.foo  # noqa
