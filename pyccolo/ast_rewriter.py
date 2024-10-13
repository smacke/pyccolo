# -*- coding: utf-8 -*-
import ast
import logging
from collections import defaultdict
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from pyccolo.ast_bookkeeping import AstBookkeeper, BookkeepingVisitor
from pyccolo.expr_rewriter import ExprRewriter
from pyccolo.handler import HandlerSpec
from pyccolo.predicate import CompositePredicate, Predicate
from pyccolo.stmt_inserter import StatementInserter
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.syntax_augmentation import AugmentationSpec, fix_positions
from pyccolo.trace_events import TraceEvent

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


_T = TypeVar("_T")
GUARD_DATA_T = Tuple[HandlerSpec, Callable[[Union[int, ast.AST]], str]]


class AstRewriter(ast.NodeTransformer):
    gc_bookkeeping = True

    def __init__(
        self,
        tracers: "List[BaseTracer]",
        path: str,
        module_id: Optional[int] = None,
    ) -> None:
        self._tracers = tracers
        self._path = path
        self._module_id = module_id
        self._augmented_positions_by_spec: Dict[
            AugmentationSpec, Set[Tuple[int, int]]
        ] = defaultdict(set)
        self.orig_to_copy_mapping: Optional[Dict[int, ast.AST]] = None

    @contextmanager
    def tracer_override_context(
        self, tracers: List["BaseTracer"], path: str
    ) -> Generator[None, None, None]:
        orig_tracers = self._tracers
        orig_path = self._path
        self._tracers = tracers
        self._path = path
        try:
            yield
        finally:
            self._tracers = orig_tracers
            self._path = orig_path

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

    def _make_node_copy_flyweight(
        self, predicate: Callable[..., _T]
    ) -> Callable[..., _T]:
        return lambda node_or_id: predicate(
            (self.orig_to_copy_mapping or {}).get(
                node_or_id if isinstance(node_or_id, int) else id(node_or_id),
                node_or_id,
            )
        )

    def should_instrument_with_tracer(self, tracer: "BaseTracer") -> bool:
        return self._path is None or tracer._should_instrument_file_impl(self._path)

    def visit(self, node: ast.AST):
        assert isinstance(
            node, (ast.Expression, ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        assert self._path is not None
        mapper = StatementMapper(
            self._tracers,
            fix_positions(
                self._augmented_positions_by_spec,
                spec_order=self._get_order_of_specs_applied(),
            ),
        )
        orig_to_copy_mapping = mapper(node)
        last_tracer = self._tracers[-1]
        old_bookkeeper = last_tracer.ast_bookkeeper_by_fname.get(self._path)
        module_id = id(node) if self._module_id is None else self._module_id
        new_bookkeeper = last_tracer.ast_bookkeeper_by_fname[
            self._path
        ] = AstBookkeeper.create(self._path, module_id)
        if old_bookkeeper is not None and self.gc_bookkeeping:
            last_tracer.remove_bookkeeping(old_bookkeeper, module_id)
        BookkeepingVisitor(
            new_bookkeeper.ast_node_by_id,
            new_bookkeeper.containing_ast_by_id,
            new_bookkeeper.containing_stmt_by_id,
            new_bookkeeper.parent_stmt_by_id,
            new_bookkeeper.stmt_by_lineno,
        ).visit(orig_to_copy_mapping[id(node)])
        last_tracer.add_bookkeeping(new_bookkeeper, module_id)
        self.orig_to_copy_mapping = orig_to_copy_mapping
        raw_handler_predicates_by_event: DefaultDict[
            TraceEvent, List[Predicate]
        ] = defaultdict(list)
        raw_guard_exempt_handler_predicates_by_event: DefaultDict[
            TraceEvent, List[Predicate]
        ] = defaultdict(list)

        for tracer in self._tracers:
            if not self.should_instrument_with_tracer(tracer):
                continue
            for evt in tracer.events_with_registered_handlers:
                # this is to deal with the tests in test_trace_events.py,
                # which patch events_with_registered_handlers but not _event_handlers
                handler_data = tracer._event_handlers.get(
                    evt, [HandlerSpec.empty()]  # type: ignore
                )
                for handler_spec in handler_data:
                    raw_handler_predicates_by_event[evt].append(handler_spec.predicate)
                    if handler_spec.exempt_from_guards:
                        raw_guard_exempt_handler_predicates_by_event[evt].append(
                            handler_spec.predicate
                        )
        handler_predicate_by_event: DefaultDict[
            TraceEvent, Callable[..., bool]
        ] = defaultdict(
            lambda: (lambda *_: False)  # type: ignore
        )
        guard_exempt_handler_prediate_by_event: DefaultDict[
            TraceEvent, Callable[..., bool]
        ] = defaultdict(
            lambda: (lambda *_: False)  # type: ignore
        )
        for evt, raw_predicates in raw_handler_predicates_by_event.items():
            handler_predicate_by_event[evt] = self._make_node_copy_flyweight(
                CompositePredicate.any(raw_predicates)
            )
        for evt, raw_predicates in raw_guard_exempt_handler_predicates_by_event.items():
            guard_exempt_handler_prediate_by_event[
                evt
            ] = self._make_node_copy_flyweight(CompositePredicate.any(raw_predicates))
        handler_guards_by_event: DefaultDict[
            TraceEvent, List[GUARD_DATA_T]
        ] = defaultdict(list)
        for tracer in self._tracers:
            for evt, handler_specs in tracer._event_handlers.items():
                handler_guards_by_event[evt].extend(
                    (spec, self._make_node_copy_flyweight(spec.guard))
                    for spec in handler_specs
                    if spec.guard is not None
                )
        if isinstance(node, ast.Module):
            for tracer in self._tracers:
                tracer._static_init_module_impl(
                    orig_to_copy_mapping.get(id(node), node)  # type: ignore
                )
        # very important that the eavesdropper does not create new ast nodes for ast.stmt (but just
        # modifies existing ones), since StatementInserter relies on being able to map these
        expr_rewriter = ExprRewriter(
            self._tracers,
            mapper,
            orig_to_copy_mapping,
            handler_predicate_by_event,
            guard_exempt_handler_prediate_by_event,
            handler_guards_by_event,
        )
        if isinstance(node, ast.Expression):
            node = expr_rewriter.visit(node)
        else:
            for i in range(len(node.body)):
                node.body[i] = expr_rewriter.visit(node.body[i])
            node = StatementInserter(
                self._tracers,
                mapper,
                orig_to_copy_mapping,
                handler_predicate_by_event,
                guard_exempt_handler_prediate_by_event,
                handler_guards_by_event,
                expr_rewriter,
            ).visit(node)
        if not any(tracer.requires_ast_bookkeeping for tracer in self._tracers):
            last_tracer.remove_bookkeeping(new_bookkeeper, module_id)
        return node
