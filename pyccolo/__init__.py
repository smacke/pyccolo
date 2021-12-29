# -*- coding: utf-8 -*-
"""
Pyccolo: embedded instrumentation for Python.

Pyccolo brings metaprogramming to everybody via general
event-emitting AST transformations.
"""
import ast
import functools
import inspect
import textwrap
import types
from contextlib import contextmanager
from typing import Any, Dict
from .ast_rewriter import AstRewriter
from .emit_event import _TRACER_STACK, allow_reentrant_event_handling
from .extra_builtins import EMIT_EVENT, TRACING_ENABLED, make_guard_name
from .expr_rewriter import ExprRewriter
from .stmt_inserter import StatementInserter
from .stmt_mapper import StatementMapper
from .syntax_augmentation import (
    AUGMENTED_SYNTAX_REGEX_TEMPLATE,
    AugmentationSpec,
    AugmentationType,
    replace_tokens_and_get_augmented_positions,
)
from .trace_events import TraceEvent
from .trace_events import *
from .trace_stack import TraceStack
from .tracer import (
    BaseTracer,
    Null,
    register_handler,
    register_raw_handler,
    skip_when_tracing_disabled,
)
from .utils import multi_context


# convenience functions for managing tracer singleton
def tracer() -> BaseTracer:
    if len(_TRACER_STACK) > 0:
        return _TRACER_STACK[-1]
    else:
        return BaseTracer()


def instance() -> BaseTracer:
    return tracer()


def parse(code: str) -> ast.Module:
    return tracer().parse(code)


def exec(code: str, *args, **kwargs) -> Dict[str, Any]:
    return tracer().exec(code, *args, **kwargs)


def instrumented(tracers):
    def decorator(f):
        f_defined_file = f.__code__.co_filename
        with multi_context([tracer.tracing_disabled() for tracer in tracers]):
            code = ast.parse(textwrap.dedent(inspect.getsource(f)))
            code.body[0] = tracers[-1].make_ast_rewriter().visit(code.body[0])
            compiled: types.CodeType = compile(code, f.__code__.co_filename, "exec")
            for const in compiled.co_consts:
                if (
                    isinstance(const, types.CodeType)
                    and const.co_name == f.__code__.co_name
                ):
                    f.__code__ = const
                    break

        @functools.wraps(f)
        def instrumented_f(*args, **kwargs):
            with multi_context(
                [
                    tracer.tracing_enabled(tracing_enabled_file=f_defined_file)
                    for tracer in tracers
                ]
            ):
                return f(*args, **kwargs)

        return instrumented_f

    return decorator


@contextmanager
def tracing_context(tracers, *args, **kwargs):
    with multi_context([tracer.tracing_context(*args, **kwargs) for tracer in tracers]):
        yield


@contextmanager
def tracing_enabled(tracers, *args, **kwargs):
    with multi_context([tracer.tracing_enabled(*args, **kwargs) for tracer in tracers]):
        yield


@contextmanager
def tracing_disabled(tracers, *args, **kwargs):
    with multi_context(
        [tracer.tracing_disabled(*args, **kwargs) for tracer in tracers]
    ):
        yield


# redundant; do this just in case we forgot to add stubs in trace_events.py
for evt in TraceEvent:
    globals()[evt.name] = evt

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
