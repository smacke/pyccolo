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


SkipAll = object()


def _emit_tracer_loop(
    event,
    node_id,
    frame,
    kwargs,
):
    global _allow_reentrant_event_handling
    global _allow_event_handling
    orig_allow_reentrant_event_handling = _allow_reentrant_event_handling
    is_reentrant = not _allow_event_handling
    reentrant_handlers_only = is_reentrant and not _allow_reentrant_event_handling
    _allow_event_handling = False
    for tracer in _TRACER_STACK:
        _allow_reentrant_event_handling = False
        if not tracer._file_passes_filter_impl(
            event, frame.f_code.co_filename, is_reentrant=is_reentrant
        ):
            continue
        _allow_reentrant_event_handling = orig_allow_reentrant_event_handling
        new_ret = tracer._emit_event(
            event,
            node_id,
            frame,
            reentrant_handlers_only=reentrant_handlers_only,
            **kwargs,
        )
        if isinstance(new_ret, tuple) and len(new_ret) > 1 and new_ret[0] is SkipAll:
            kwargs["ret"] = new_ret[1]
            break
        else:
            kwargs["ret"] = new_ret


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
        _emit_tracer_loop(
            event,
            node_id,
            frame,
            kwargs,
        )
    finally:
        _allow_event_handling = orig_allow_event_handling
        _allow_reentrant_event_handling = orig_allow_reentrant_event_handling
    return _make_ret(event, kwargs.get("ret", None))
