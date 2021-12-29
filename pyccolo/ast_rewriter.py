# -*- coding: utf-8 -*-
import ast
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from pyccolo.expr_rewriter import ExprRewriter
from pyccolo.stmt_inserter import StatementInserter
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent
from pyccolo.syntax_augmentation import AugmentationSpec

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class AstRewriter(ast.NodeTransformer):
    def __init__(
        self,
        tracers: "List[BaseTracer]",
        module_id: Optional[int] = None,
        path: Optional[str] = None,
    ):
        self._tracers = tracers
        self._module_id: Optional[int] = module_id
        self._path: Optional[str] = path
        self._augmented_positions_by_spec: Dict[
            AugmentationSpec, Set[Tuple[int, int]]
        ] = defaultdict(set)
        self.orig_to_copy_mapping: Optional[Dict[int, ast.AST]] = None

    def register_augmented_position(
        self, aug_spec: AugmentationSpec, lineno: int, col_offset: int
    ) -> None:
        self._augmented_positions_by_spec[aug_spec].add((lineno, col_offset))

    def visit(self, node: ast.AST):
        assert isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef))
        mapper = StatementMapper(
            self._tracers[-1].line_to_stmt_by_module_id[
                id(node) if self._module_id is None else self._module_id
            ],
            self._tracers,
            self._augmented_positions_by_spec,
        )
        orig_to_copy_mapping = mapper(node)
        self.orig_to_copy_mapping = orig_to_copy_mapping
        # very important that the eavesdropper does not create new ast nodes for ast.stmt (but just
        # modifies existing ones), since StatementInserter relies on being able to map these
        events_with_handlers: Set[TraceEvent] = set()
        for tracer in self._tracers:
            for evt in tracer.events_with_registered_handlers:
                if self._path is None or tracer._file_passes_filter_impl(
                    evt.value, self._path
                ):
                    events_with_handlers.add(evt)
        frozen_events_with_handlers = frozenset(events_with_handlers)
        expr_rewriter = ExprRewriter(
            orig_to_copy_mapping, frozen_events_with_handlers, self._tracers[-1].guards
        )
        for i in range(len(node.body)):
            node.body[i] = expr_rewriter.visit(node.body[i])
        node = StatementInserter(
            orig_to_copy_mapping, frozen_events_with_handlers, self._tracers[-1].guards
        ).visit(node)
        return node
