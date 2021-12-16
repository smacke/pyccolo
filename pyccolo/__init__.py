# -*- coding: future_annotations -*-
from .tracing.ast_rewriter import AstRewriter
from .tracing.extra_builtins import EMIT_EVENT, TRACING_ENABLED, make_guard_name
from .tracing.expr_rewriter import ExprRewriter
from .tracing.stmt_inserter import StatementInserter
from .tracing.stmt_mapper import StatementMapper
from .tracing.syntax_augmentation import (
	AUGMENTED_SYNTAX_REGEX_TEMPLATE,
	AugmentationSpec,
	AugmentationType,
	replace_tokens_and_get_augmented_positions,
)
from .tracing.trace_events import TraceEvent
from .tracing.trace_stack import TraceStack
from .tracing.tracer import (
	BaseTracerStateMachine,
	register_handler,
	register_raw_handler,
	skip_when_tracing_disabled,
)

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
