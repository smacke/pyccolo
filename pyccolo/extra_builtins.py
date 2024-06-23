# -*- coding: utf-8 -*-
import ast
from typing import Union

PYCCOLO_BUILTIN_PREFIX = "_X5ix"
EMIT_EVENT = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_EVT_EMIT"
TRACE_LAMBDA = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_TRACE_LAM"
EXEC_SAVED_THUNK = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_EXEC_SAVED_THUNK"
TRACING_ENABLED = f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_TRACING_ENABLED"


def make_guard_name(node: Union[int, ast.AST]):
    node_id = node if isinstance(node, int) else id(node)
    return f"{PYCCOLO_BUILTIN_PREFIX}_PYCCOLO_GUARD_{node_id}"
