# -*- coding: utf-8 -*-
import ast
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, TypeVar

from pyccolo import fast
from pyccolo.emit_event import _TRACER_STACK

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


_T = TypeVar("_T", bound=ast.AST)


class StatementMapper(ast.NodeVisitor):
    def __init__(self, tracers: Optional[List["BaseTracer"]] = None):
        self._tracers: List["BaseTracer"] = (
            _TRACER_STACK if tracers is None else tracers
        )
        self.traversal: List[ast.AST] = []

    @classmethod
    def bookkeeping_propagating_copy(cls, node: _T) -> _T:
        return cls()(node)[id(node)]  # type: ignore[return-value]

    def _handle_augmentations(self, no: ast.AST, nc: ast.AST) -> None:
        for tracer in self._tracers:
            augs = tracer.get_augmentations(id(no))
            if augs:
                # ``augmented_node_ids_by_spec`` is keyed by ``id(nc)``, so a
                # registered entry is only meaningful while ``nc`` stays alive;
                # otherwise the copy could be garbage collected and its address
                # recycled by an unrelated node, yielding a spurious hit (e.g. a
                # plain name mistaken for a ``$`` placeholder). Pin the copy so
                # its id cannot be reused while it remains registered. Copies
                # produced during a parse are additionally bookkept (and thus gc'd
                # together) by the ast rewriter; runtime copies made via
                # ``bookkeeping_propagating_copy`` would otherwise leak a dangling
                # id here.
                tracer.ast_node_by_id[id(nc)] = nc
            for spec in augs:
                tracer.augmented_node_ids_by_spec[spec].add(id(nc))

    def __call__(
        self,
        node: ast.AST,
        copy_node: Optional[ast.AST] = None,
    ) -> Dict[int, ast.AST]:
        # for some bizarre reason we need to visit once to clear empty nodes apparently
        self.traversal.clear()
        self.visit(node)
        self.traversal.clear()

        self.visit(node)
        orig_traversal = self.traversal
        self.traversal = []
        self.visit(copy_node or fast.copy_ast(node))
        copy_traversal = self.traversal
        orig_to_copy_mapping = {}
        if len(self._tracers) > 0:
            tracer = self._tracers[-1]
        else:
            tracer = None
        for no, nc in zip(orig_traversal, copy_traversal):
            orig_to_copy_mapping[id(no)] = nc
            if hasattr(nc, "lineno"):
                self._handle_augmentations(no, nc)
            if tracer is None:
                continue
            for extra_bookkeeping in tracer.additional_ast_bookkeeping.values():
                if id(no) not in extra_bookkeeping:
                    continue
                if isinstance(extra_bookkeeping, set):
                    extra_bookkeeping.add(id(nc))
                else:
                    extra_bookkeeping[id(nc)] = extra_bookkeeping[id(no)]
        return orig_to_copy_mapping

    def visit(self, node: ast.AST) -> None:
        self.traversal.append(node)
        for name, field in ast.iter_fields(node):
            if isinstance(field, ast.AST):
                self.visit(field)
            elif isinstance(field, list):
                for inner_node in field:
                    if isinstance(inner_node, ast.AST):
                        self.visit(inner_node)
