# -*- coding: future_annotations -*-
"""
Pyccolo: embedded instrumentation for Python.

Pyccolo brings metaprogramming to everybody via general
event-emitting AST transformations.
"""
import ast
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


# redundant; do this just in case we forgot to add stubs in trace_events.py
for evt in TraceEvent:
	globals()[evt.name] = evt

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
