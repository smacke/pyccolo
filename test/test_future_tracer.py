# -*- coding: utf-8 -*-
import pyccolo as pyc
from pyccolo.examples import FutureTracer


def test_simple():
    with FutureTracer.instance():
        pyc.exec(
            """
            def foo():
                return 0
            x = foo()
            y = x + 1
            z = y + 2
            assert y == 1, "got %s" % y
            assert z == 3, "got %s" % z
            """
        )
    FutureTracer.clear_instance()
    assert not any(isinstance(tracer, FutureTracer) for tracer in pyc._TRACER_STACK), (
        "got %s" % pyc._TRACER_STACK
    )


if __name__ == "__main__":
    test_simple()
