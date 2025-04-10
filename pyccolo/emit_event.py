# -*- coding: utf-8 -*-
import logging
import sys
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, List

from pyccolo.trace_events import BEFORE_EXPR_EVENTS

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer, _InternalBaseTracer


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
_main_thread_id = threading.main_thread().ident
SANDBOX_FNAME = "<sandbox>"
SANDBOX_FNAME_PREFIX = "<sandbox"


def _should_instrument_file(tracer: "_InternalBaseTracer", filename: str) -> bool:
    return False


def _should_instrument_file_impl(tracer, filename: str) -> bool:
    if (
        tracer.instrument_all_files
        or filename in tracer._tracing_enabled_files
        or filename.startswith(SANDBOX_FNAME_PREFIX)
    ):
        return True
    for clazz in tracer.__class__.mro():
        if clazz.__name__ == "BaseTracer":
            break
        should_instrument_file = clazz.__dict__.get("should_instrument_file")
        if should_instrument_file is not None:
            return should_instrument_file(tracer, filename)
    return _should_instrument_file(tracer, filename)


def _file_passes_filter_for_event(
    tracer: "_InternalBaseTracer", evt: str, filename: str
) -> bool:
    return True


def _file_passes_filter_impl(
    tracer: "_InternalBaseTracer", evt: str, filename: str, is_reentrant: bool = False
) -> bool:
    if filename == tracer._current_sandbox_fname and tracer.has_sys_trace_events:
        ret = tracer._num_sandbox_calls_seen >= 2
        tracer._num_sandbox_calls_seen += evt == "call"
        return ret
    if not (
        evt
        in (
            "before_import",
            "init_module",
            "after_import",
        )
        or _should_instrument_file_impl(tracer, filename)
    ):
        return False
    for clazz in tracer.__class__.mro():
        if clazz.__name__ == "BaseTracer":
            break
        file_passes_filter_for_event = clazz.__dict__.get(
            "file_passes_filter_for_event"
        )
        if file_passes_filter_for_event is not None:
            return file_passes_filter_for_event(tracer, evt, filename)
    return _file_passes_filter_for_event(tracer, evt, filename)


def _emit_tracer_loop(
    event,
    node_id,
    frame,
    kwargs,
):
    global _allow_reentrant_event_handling
    global _allow_event_handling
    current_thread_id = threading.current_thread().ident
    is_reentrant = not _allow_event_handling
    reentrant_handlers_only = is_reentrant and not _allow_reentrant_event_handling
    _allow_event_handling = False
    for tracer in _TRACER_STACK:
        if current_thread_id != _main_thread_id and not tracer.multiple_threads_allowed:
            continue
        if (
            is_reentrant
            and not tracer.allow_reentrant_events
            and not _allow_reentrant_event_handling
        ):
            continue
        if not _file_passes_filter_impl(
            tracer, event, frame.f_code.co_filename, is_reentrant=is_reentrant
        ):
            continue
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
    __debuggerskip__ = True  # noqa: F841
    frame = sys._getframe().f_back
    if frame.f_code.co_filename == __file__:
        # weird shit happens if we instrument this file, so exclude it.
        return _make_ret(event, kwargs.get("ret", None))
    orig_allow_event_handling = _allow_event_handling
    orig_allow_reentrant_event_handling = _allow_reentrant_event_handling
    if len(_TRACER_STACK) > 0:
        remapping = _TRACER_STACK[-1].node_id_remapping_by_fname.get(
            frame.f_code.co_filename
        )
        if remapping is not None:
            node_id = remapping.get(node_id, node_id)
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
    return _make_ret(event, kwargs.get("ret"))
