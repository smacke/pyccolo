# -*- coding: utf-8 -*-
import ast
import bisect
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
from pyccolo.syntax_augmentation import (
    AugmentationSpec,
    AugmentationType,
    Position,
    Range,
    fix_positions,
)
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
        self._augmented_positions_by_spec: Dict[AugmentationSpec, Set[Position]] = (
            defaultdict(set)
        )
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
            for spec in tracer.syntax_augmentation_specs():
                if spec not in specs:
                    specs.append(spec)
        return tuple(specs)

    def register_augmented_position(
        self, aug_spec: AugmentationSpec, lineno: int, col_offset: int
    ) -> None:
        self._augmented_positions_by_spec[aug_spec].add(Position(lineno, col_offset))

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

    @staticmethod
    def _get_prefix_range_for(
        node: ast.AST,
    ) -> Optional[Range]:
        line, col = None, None
        if isinstance(node, ast.Name):
            line, col = node.lineno, node.col_offset
        elif isinstance(node, ast.Attribute):
            line, col = node.lineno, getattr(node.value, "end_col_offset", -2) + 1
        elif isinstance(node, ast.FunctionDef):
            # TODO: can be different if more spaces between 'def' and function name
            line, col = node.lineno, node.col_offset + 4
        elif isinstance(node, ast.ClassDef):
            # TODO: can be different if more spaces between 'class' and class name
            line, col = node.lineno, node.col_offset + 6
        elif isinstance(node, ast.AsyncFunctionDef):
            # TODO: can be different if more spaces between 'async', 'def', and function name
            line, col = node.lineno, node.col_offset + 10
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and len(node.names) == 1:
            # "import " vs "from <base_module> import "
            base_offset = (
                7 if isinstance(node, ast.Import) else 13 + len(node.module or "")
            )
            name = node.names[0]
            line, col = node.lineno, (
                node.col_offset
                + base_offset
                + (0 if name.asname is None else len(name.name) + 1)
            )
        if line is None or col is None:
            return None
        else:
            return Range.singleton_span(line, col)

    @staticmethod
    def _get_suffix_range_for(
        node: ast.AST,
    ) -> Optional[Range]:
        line, col = None, None
        if isinstance(node, ast.Name):
            line, col = node.lineno, node.col_offset + len(node.id)
        elif isinstance(node, ast.Attribute):
            line, col = (
                node.lineno,
                getattr(node.value, "end_col_offset", -1) + len(node.attr) + 1,
            )
        elif isinstance(node, ast.FunctionDef):
            # TODO: can be different if more spaces between 'def' and function name
            line, col = node.lineno, node.col_offset + 4 + len(node.name)
        elif isinstance(node, ast.ClassDef):
            # TODO: can be different if more spaces between 'class' and class name
            line, col = node.lineno, node.col_offset + 6 + len(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            # TODO: can be different if more spaces between 'async', 'def', and function name
            line, col = node.lineno, node.col_offset + 10 + len(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and len(node.names) == 1:
            name = node.names[0]
            # "import " vs "from <base_module> import "
            base_offset = (
                7 if isinstance(node, ast.Import) else 13 + len(node.module or "")
            )
            col_offset = node.col_offset + base_offset
            if name.asname is None:
                col_offset += len(name.name)
            else:
                col_offset += len(name.name) + 1 + len(name.asname)
            line, col = node.lineno, col_offset
        if line is None or col is None:
            return None
        else:
            return Range.singleton_span(line, col)

    @staticmethod
    def _get_dot_suffix_range_for(
        node: ast.AST,
    ) -> Optional[Range]:
        end_lineno, end_col_offset = None, None
        if isinstance(node, ast.Name):
            end_lineno = getattr(node, "end_lineno", None)
            end_col_offset = getattr(node, "end_col_offset", None)
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            end_lineno = getattr(node.value, "end_lineno", None)
            end_col_offset = getattr(node.value, "end_col_offset", None)
        if end_lineno is None or end_col_offset is None:
            return None
        else:
            return Range.singleton_span(end_lineno, end_col_offset)

    @staticmethod
    def _get_dot_prefix_range_for(
        node: ast.AST,
    ) -> Optional[Range]:
        line, col = None, None
        if isinstance(node, ast.Name):
            line, col = node.lineno, node.col_offset
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            line, col = node.value.lineno, node.value.col_offset
        if line is None or col is None:
            return None
        else:
            return Range.singleton_span(line, col)

    @staticmethod
    def _get_binop_range_for(
        node: ast.AST,
    ) -> Optional[Range]:
        if isinstance(node, ast.BinOp):
            left_end_lineno: Optional[int] = getattr(node.left, "end_lineno", None)
            left_end_col_offset: Optional[int] = getattr(
                node.left, "end_col_offset", None
            )
            if left_end_lineno is None or left_end_col_offset is None:
                return None
            else:
                return Range(
                    Position(left_end_lineno, left_end_col_offset),
                    Position(node.right.lineno, node.right.col_offset),
                )
        else:
            return None

    def _get_boolop_range_for(self, node: ast.AST) -> Optional[Range]:
        if not hasattr(node, "col_offset") or not isinstance(node, ast.expr):
            return None
        parent = self._tracers[-1].containing_ast_by_id.get(id(node))
        if not isinstance(parent, ast.BoolOp):
            return None
        try:
            value_index = parent.values.index(node)
            if value_index + 1 >= len(parent.values):
                return None
        except ValueError:
            return None
        sibling = parent.values[value_index + 1]
        end_lineno = getattr(sibling, "end_lineno", None)
        end_col_offset = getattr(sibling, "end_col_offset", None)
        if end_lineno is None or end_col_offset is None:
            return None
        return Range(
            Position(node.lineno, node.col_offset), Position(end_lineno, end_col_offset)
        )

    def _get_call_range_for(self, node: ast.AST) -> Optional[Range]:
        if not isinstance(node, ast.Call):
            return None
        end_lineno: Optional[int] = getattr(node.func, "end_lineno", None)
        end_col_offset: Optional[int] = getattr(node.func, "end_col_offset", None)
        if end_lineno is None or end_col_offset is None:
            return None
        return Range.singleton_span(end_lineno, end_col_offset)

    def _get_range_for(
        self, aug_type: AugmentationType, node: ast.AST
    ) -> Optional[Range]:
        if aug_type == AugmentationType.prefix:
            return self._get_prefix_range_for(node)
        elif aug_type == AugmentationType.suffix:
            return self._get_suffix_range_for(node)
        elif aug_type == AugmentationType.dot_suffix:
            return self._get_dot_suffix_range_for(node)
        elif aug_type == AugmentationType.dot_prefix:
            return self._get_dot_prefix_range_for(node)
        elif aug_type == AugmentationType.binop:
            return self._get_binop_range_for(node)
        elif aug_type == AugmentationType.boolop:
            return self._get_boolop_range_for(node)
        elif aug_type == AugmentationType.call:
            return self._get_call_range_for(node)
        else:
            raise NotImplementedError()

    def _handle_augmentations_for_node(
        self,
        augmented_positions_by_spec: Dict[AugmentationSpec, List[Position]],
        nc: ast.AST,
    ) -> None:
        for spec, mod_positions in augmented_positions_by_spec.items():
            aug_range = self._get_range_for(spec.aug_type, nc)
            if aug_range is None:
                continue
            start_pos, end_pos = aug_range
            left_insert_point = bisect.bisect_left(mod_positions, start_pos)
            right_insert_point = bisect.bisect_right(mod_positions, end_pos)
            if not any(
                start_pos <= mod_positions[i] <= end_pos
                for i in range(
                    max(left_insert_point, 0),
                    min(right_insert_point, len(mod_positions)),
                )
            ):
                continue
            for tracer in self._tracers:
                if spec in tracer.syntax_augmentation_specs():
                    tracer.augmented_node_ids_by_spec[spec].add(id(nc))

    def _handle_all_augmentations(
        self, orig_to_copy_mapping: Dict[int, ast.AST]
    ) -> None:
        augmented_positions_by_spec = fix_positions(
            self._augmented_positions_by_spec,
            spec_order=self._get_order_of_specs_applied(),
        )
        for nc in orig_to_copy_mapping.values():
            self._handle_augmentations_for_node(augmented_positions_by_spec, nc)

    def visit(self, node: ast.AST):
        assert isinstance(
            node, (ast.Expression, ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)
        )
        assert self._path is not None
        mapper = StatementMapper(self._tracers)
        orig_to_copy_mapping = mapper(node)
        last_tracer = self._tracers[-1]
        old_bookkeeper = last_tracer.ast_bookkeeper_by_fname.get(self._path)
        module_id = id(node) if self._module_id is None else self._module_id

        # garbage collect any stale references to aug specs once they have been propagated
        cleanup_bookkeeper = AstBookkeeper.create(self._path, module_id)
        BookkeepingVisitor(cleanup_bookkeeper).visit(node)
        last_tracer.remove_bookkeeping(cleanup_bookkeeper, module_id)

        new_bookkeeper = last_tracer.ast_bookkeeper_by_fname[self._path] = (
            AstBookkeeper.create(self._path, module_id)
        )
        if old_bookkeeper is not None and self.gc_bookkeeping:
            last_tracer.remove_bookkeeping(old_bookkeeper, module_id)
        BookkeepingVisitor(new_bookkeeper).visit(orig_to_copy_mapping[id(node)])
        last_tracer.add_bookkeeping(new_bookkeeper, module_id)
        self.orig_to_copy_mapping = orig_to_copy_mapping
        self._handle_all_augmentations(orig_to_copy_mapping)
        raw_handler_predicates_by_event: DefaultDict[TraceEvent, List[Predicate]] = (
            defaultdict(list)
        )
        raw_guard_exempt_handler_predicates_by_event: DefaultDict[
            TraceEvent, List[Predicate]
        ] = defaultdict(list)

        for tracer in self._tracers:
            if not self.should_instrument_with_tracer(tracer):
                continue
            for evt in tracer.events_with_registered_handlers:
                # this is to deal with the tests in test_trace_events.py,
                # which patch events_with_registered_handlers but do not add them to _event_handlers
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
            guard_exempt_handler_prediate_by_event[evt] = (
                self._make_node_copy_flyweight(CompositePredicate.any(raw_predicates))
            )
        handler_guards_by_event: DefaultDict[TraceEvent, List[GUARD_DATA_T]] = (
            defaultdict(list)
        )
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
