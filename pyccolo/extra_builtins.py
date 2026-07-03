# -*- coding: utf-8 -*-
import ast
import re
from typing import Union

PYCCOLO_BUILTIN_PREFIX = "_X5ix"
EMIT_EVENT = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_EVT_EMIT"
TRACE_LAMBDA = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_TRACE_LAM"
EXEC_SAVED_THUNK = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_EXEC_SAVED_THUNK"
TRACING_ENABLED = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_TRACING_ENABLED"
FUNCTION_TRACING_ENABLED = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_FUNCTION_TRACING_ENABLED"
NAME_ERROR_MATCHES = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_NAME_ERR_MATCHES"

_NAME_ERROR_RE = re.compile(r"name '([^']+)' is not defined")


def name_error_matches_prefix(exc: BaseException, prefix: str) -> bool:
    """Whether the undefined name referenced by ``exc`` starts with ``prefix``.

    Used by the global-guards retry to distinguish a missing pyccolo builtin
    (which should trigger a globals-copy retry) from a genuine user-level
    ``NameError`` (which must propagate). ``NameError.name`` only exists on
    Python 3.10+, so fall back to parsing the message on older versions. An
    ``UnboundLocalError`` (a ``NameError`` subclass whose message does not match
    the "name '...' is not defined" shape) therefore yields ``False`` and is
    re-raised, rather than crashing on a missing ``.name`` attribute."""
    name = getattr(exc, "name", None)
    if name is None:
        match = _NAME_ERROR_RE.search(str(exc))
        if match is None:
            return False
        name = match.group(1)
    return name.startswith(prefix)


def make_guard_name(node: Union[int, ast.AST]):
    node_id = node if isinstance(node, int) else id(node)
    return f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_GUARD_{node_id}"
