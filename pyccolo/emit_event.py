# -*- coding: future_annotations -*-
import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import List
    from pyccolo.tracer import BaseTracerStateMachine


_TRACER_STACK: List[BaseTracerStateMachine] = []
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


def _emit_event(event, node_id, **kwargs):
    global _allow_event_handling
    if _allow_event_handling or _allow_reentrant_event_handling:
        _allow_event_handling = False
        frame = sys._getframe().f_back
        try:
            for tracer in _TRACER_STACK:
                kwargs['ret'] = tracer._emit_event(event, node_id, frame, **kwargs)
        finally:
            _allow_event_handling = True
    return kwargs.get('ret', None)
