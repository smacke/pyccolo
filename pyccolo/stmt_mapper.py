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
    def augmentation_propagating_copy(cls, node: _T) -> _T:
        return cls()(node)[id(node)]  # type: ignore[return-value]

    def _handle_augmentations(self, no: ast.AST, nc: ast.AST) -> None:
        for tracer in self._tracers:
            for spec in tracer.get_augmentations(id(no)):
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
        for no, nc in zip(orig_traversal, copy_traversal):
            orig_to_copy_mapping[id(no)] = nc
            if hasattr(nc, "lineno"):
                self._handle_augmentations(no, nc)
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
