# -*- coding: future_annotations -*-
from .ast_rewriter import AstRewriter
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
from .trace_events import *
from .trace_stack import TraceStack
from .tracer import (
	BaseTracerStateMachine,
	Null,
	register_handler,
	register_raw_handler,
	skip_when_tracing_disabled,
)


# convenience functions for managing tracer singleton
def tracer() -> BaseTracerStateMachine:
	return BaseTracerStateMachine.instance()


def instance() -> BaseTracerStateMachine:
	return tracer()


def exec_raw(*args, **kwargs):
	return tracer().exec_raw(*args, **kwargs)


def exec(*args, **kwargs):
	return tracer().exec(*args, **kwargs)


def clear_instance() -> None:
	return BaseTracerStateMachine.clear_instance()

# redundant; do this just in case we forgot to add stubs in trace_events.py
for evt in TraceEvent:
	globals()[evt.name] = evt

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
