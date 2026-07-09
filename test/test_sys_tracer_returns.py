# -*- coding: utf-8 -*-
"""Whatever ``_sys_tracer`` returns for a non-``call`` event becomes the frame's
local trace function, so it must be callable or None.

When no handler consumes an event, ``_emit_event`` threads its ``ret`` kwarg --
i.e. sys.settrace's ``arg`` -- straight back out. For ``exception`` that is the
``(type, value, tb)`` triple and for ``return`` it is the returned value. Handing
either to CPython used to raise ``TypeError: 'tuple' object is not callable`` on
the frame's next event.
"""
import pytest

import pyccolo as pyc


class CallOnlyTracer(pyc.BaseTracer):
    """Registers a sys-trace event, but nothing that consumes ``exception``."""

    def should_instrument_file(self, filename: str) -> bool:
        return filename.endswith("raises_and_yields.py")

    @pyc.register_handler(pyc.call)
    def handle_call(self, *_, **__):
        return None


def test_exception_in_traced_frame_propagates():
    with CallOnlyTracer.instance().tracing_enabled():
        from test.raises_and_yields import boom

        with pytest.raises(ValueError, match="boom"):
            boom()


def test_generator_resume_after_yield_in_traced_frame():
    # A yield emits a ``return`` event carrying the yielded value; resuming the
    # generator then delivers another event to the *same* frame, which is where a
    # non-callable local trace function blows up.
    with CallOnlyTracer.instance().tracing_enabled():
        from test.raises_and_yields import gen

        assert list(gen()) == [1, 2]
