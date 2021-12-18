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
from .trace_events import TraceEvent
from .trace_stack import TraceStack
from .tracer import (
	BaseTracerStateMachine,
	Null,
	register_handler,
	register_raw_handler,
	skip_when_tracing_disabled,
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
