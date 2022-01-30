# -*- coding: utf-8 -*-
import ast
from typing import Union


EMIT_EVENT = "_X5ix_PYCCOLO_EVT_EMIT"
TRACE_LAMBDA = "X5ix_PYCCOLO_TRACE_LAM"
EXEC_SAVED_THUNK = "_X5ix_PYCCOLO_EXEC_SAVED_THUNK"
TRACING_ENABLED = "_X5ix_PYCCOLO_TRACING_ENABLED"


def make_guard_name(node: Union[int, ast.AST]):
    node_id = node if isinstance(node, int) else id(node)
    return "_X5ix_PYCCOLO_GUARD_{}".format(node_id)
