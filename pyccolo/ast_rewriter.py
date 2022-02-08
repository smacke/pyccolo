# -*- coding: utf-8 -*-
import ast
import logging
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

from pyccolo.ast_bookkeeping import BookkeepingVisitor
from pyccolo.expr_rewriter import ExprRewriter
from pyccolo.predicate import CompositePredicate, Predicate
from pyccolo.stmt_inserter import StatementInserter
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent
from pyccolo.syntax_augmentation import AugmentationSpec, fix_positions

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

    def _get_order_of_specs_applied(self) -> Tuple[AugmentationSpec, ...]:
        specs = []
        for tracer in self._tracers:
            for spec in tracer.syntax_augmentation_specs:
                if spec not in specs:
                    specs.append(spec)
        return tuple(specs)

    def register_augmented_position(
        self, aug_spec: AugmentationSpec, lineno: int, col_offset: int
    ) -> None:
        self._augmented_positions_by_spec[aug_spec].add((lineno, col_offset))

    def _make_node_copy_flyweight(self, predicate: Predicate) -> Callable[..., bool]:
        return lambda node: predicate(self.orig_to_copy_mapping.get(id(node), node))

    def visit(self, node: ast.AST):
        assert isinstance(
            node, (ast.Expression, ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        mapper = StatementMapper(
            self._tracers,
            fix_positions(
                self._augmented_positions_by_spec,
                spec_order=self._get_order_of_specs_applied(),
            ),
        )
        orig_to_copy_mapping = mapper(node)
        BookkeepingVisitor(
            self._tracers[-1].ast_node_by_id,
            self._tracers[-1].containing_ast_by_id,
            self._tracers[-1].containing_stmt_by_id,
            self._tracers[-1].parent_stmt_by_id,
            self._tracers[-1].stmt_by_lineno_by_module_id[
                id(node) if self._module_id is None else self._module_id
            ],
        ).visit(orig_to_copy_mapping[id(node)])
        self.orig_to_copy_mapping = orig_to_copy_mapping
        raw_handler_predicates_by_event: DefaultDict[
            TraceEvent, List[Predicate]
        ] = defaultdict(list)

        for tracer in self._tracers:
            for evt in tracer.events_with_registered_handlers:
                if self._path is not None and not tracer._file_passes_filter_impl(
                    evt.value, self._path
                ):
                    continue
                # this is to deal with the tests in test_trace_events.py,
                # which patch events_with_registered_handlers but not _event_handlers
                handler_data = tracer._event_handlers.get(
                    evt, [(None, False, False, Predicate(lambda *_: True))]  # type: ignore
                )
                for _, use_raw_node_id, __, predicate in handler_data:
                    raw_handler_predicates_by_event[evt].append(predicate)
        handler_predicate_by_event: DefaultDict[
            TraceEvent, Callable[..., bool]
        ] = defaultdict(
            lambda: (lambda *_: False)  # type: ignore
        )
        for evt, raw_predicates in raw_handler_predicates_by_event.items():
            handler_predicate_by_event[evt] = self._make_node_copy_flyweight(
                CompositePredicate.any(raw_predicates)
            )
        # very important that the eavesdropper does not create new ast nodes for ast.stmt (but just
        # modifies existing ones), since StatementInserter relies on being able to map these
        expr_rewriter = ExprRewriter(
            orig_to_copy_mapping, handler_predicate_by_event, self._tracers[-1].guards
        )
        if isinstance(node, ast.Expression):
            node = expr_rewriter.visit(node)
        else:
            for i in range(len(node.body)):
                node.body[i] = expr_rewriter.visit(node.body[i])
            node = StatementInserter(
                orig_to_copy_mapping,
                handler_predicate_by_event,
                self._tracers[-1].guards,
            ).visit(node)
        return node
