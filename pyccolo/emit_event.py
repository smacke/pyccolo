# -*- coding: utf-8 -*-
import logging
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING, List

from pyccolo.trace_events import BEFORE_EXPR_EVENTS

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer


logger = logging.getLogger(__name__)


_BEFORE_EXPR_EVENT_NAMES = {evt.value for evt in BEFORE_EXPR_EVENTS}
_TRACER_STACK: "List[BaseTracer]" = []
_allow_event_handling = True
_allow_reentrant_event_handling = False


@contextmanager
def allow_reentrant_event_handling():
    global _allow_reentrant_event_handling
    orig_allow_reentrant_handling = _allow_reentrant_event_handling
    _allow_reentrant_event_handling = True
    try:
        yield
    finally:
        _allow_reentrant_event_handling = orig_allow_reentrant_handling


def _make_ret(event, ret):
    if event in _BEFORE_EXPR_EVENT_NAMES and not callable(ret):
        return lambda *_: ret
    else:
        return ret


def _emit_event(event, node_id, **kwargs):
    global _allow_event_handling
    global _allow_reentrant_event_handling
    frame = sys._getframe().f_back
    if frame.f_code.co_filename == __file__:
        # weird shit happens if we instrument this file, so exclude it.
        return _make_ret(event, kwargs.get("ret", None))
    orig_allow_event_handling = _allow_event_handling
    orig_allow_reentrant_event_handling = _allow_reentrant_event_handling
    try:
        is_reentrant = not _allow_event_handling
        reentrant_handlers_only = is_reentrant and not _allow_reentrant_event_handling
        _allow_event_handling = False
        _allow_reentrant_event_handling = False
        for tracer in _TRACER_STACK:
            if tracer._file_passes_filter_impl(
                event, frame.f_code.co_filename, is_reentrant=is_reentrant
            ):
                try:
                    _allow_reentrant_event_handling = (
                        orig_allow_reentrant_event_handling
                    )
                    kwargs["ret"] = tracer._emit_event(
                        event,
                        node_id,
                        frame,
                        reentrant_handlers_only=reentrant_handlers_only,
                        **kwargs,
                    )
                finally:
                    _allow_reentrant_event_handling = False
    finally:
        _allow_event_handling = orig_allow_event_handling
        _allow_reentrant_event_handling = orig_allow_reentrant_event_handling
    return _make_ret(event, kwargs.get("ret", None))
