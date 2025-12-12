# -*- coding: utf-8 -*-
import ast
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from pyccolo import fast
from pyccolo.emit_event import _TRACER_STACK
from pyccolo.syntax_augmentation import AugmentationSpec, AugmentationType

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class StatementMapper(ast.NodeVisitor):
    def __init__(
        self,
        tracers: Optional[List["BaseTracer"]] = None,
        augmented_positions_by_spec: Optional[
            Dict[AugmentationSpec, Set[Tuple[int, int]]]
        ] = None,
    ):
        self._tracers: List["BaseTracer"] = (
            _TRACER_STACK if tracers is None else tracers
        )
        self.augmented_positions_by_spec: Dict[
            AugmentationSpec, Set[Tuple[int, int]]
        ] = (augmented_positions_by_spec or {})
        self.traversal: List[ast.AST] = []

    @classmethod
    def augmentation_propagating_copy(cls, node: ast.AST) -> ast.AST:
        return cls()(node)[id(node)]

    @staticmethod
    def _get_prefix_col_offset_for(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Name):
            return node.col_offset
        elif isinstance(node, ast.Attribute):
            return getattr(node.value, "end_col_offset", -2) + 1
        elif isinstance(node, ast.FunctionDef):
            # TODO: can be different if more spaces between 'def' and function name
            return node.col_offset + 4
        elif isinstance(node, ast.ClassDef):
            # TODO: can be different if more spaces between 'class' and class name
            return node.col_offset + 6
        elif isinstance(node, ast.AsyncFunctionDef):
            # TODO: can be different if more spaces between 'async', 'def', and function name
            return node.col_offset + 10
        elif isinstance(node, (ast.Import, ast.ImportFrom)) and len(node.names) == 1:
            # "import " vs "from <base_module> import "
            base_offset = (
                7 if isinstance(node, ast.Import) else 13 + len(node.module or "")
            )
            name = node.names[0]
            return (
                node.col_offset
                + base_offset
                + (0 if name.asname is None else len(name.name) + 1)
            )
        else:
            return None

    @staticmethod
    def _get_suffix_col_offset_for(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Name):
            return node.col_offset + len(node.id)
        elif isinstance(node, ast.Attribute):
            return getattr(node.value, "end_col_offset", -1) + len(node.attr) + 1
        elif isinstance(node, ast.FunctionDef):
            # TODO: can be different if more spaces between 'def' and function name
            return node.col_offset + 4 + len(node.name)
        elif isinstance(node, ast.ClassDef):
            # TODO: can be different if more spaces between 'class' and class name
            return node.col_offset + 6 + len(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            # TODO: can be different if more spaces between 'async', 'def', and function name
            return node.col_offset + 10 + len(node.name)
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
            return col_offset
        else:
            return None

    @staticmethod
    def _get_dot_suffix_col_offset_for(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Name):
            return getattr(node, "end_col_offset", -1)
        elif isinstance(node, ast.Attribute):
            return getattr(node.value, "end_col_offset", -1)
        else:
            return None

    @staticmethod
    def _get_dot_prefix_col_offset_for(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Name):
            return node.col_offset
        elif isinstance(node, ast.Attribute):
            return node.value.col_offset
        else:
            return None

    @staticmethod
    def _get_binop_col_offset_for(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.BinOp):
            left_end_col_offset = getattr(node.left, "end_col_offset", None)
            if left_end_col_offset is None:
                return -1
            else:
                return node.left.col_offset - node.col_offset + left_end_col_offset + 1
        else:
            return None

    def _get_col_offset_for(
        self, aug_type: AugmentationType, node: ast.AST
    ) -> Optional[int]:
        if aug_type == AugmentationType.prefix:
            return self._get_prefix_col_offset_for(node)
        elif aug_type == AugmentationType.suffix:
            return self._get_suffix_col_offset_for(node)
        elif aug_type == AugmentationType.dot_suffix:
            return self._get_dot_suffix_col_offset_for(node)
        elif aug_type == AugmentationType.dot_prefix:
            return self._get_dot_prefix_col_offset_for(node)
        elif aug_type == AugmentationType.binop:
            return self._get_binop_col_offset_for(node)
        else:
            raise NotImplementedError()

    def _handle_augmentations(self, no: ast.AST, nc: ast.AST) -> None:
        for spec, mod_positions in self.augmented_positions_by_spec.items():
            col_offset = self._get_col_offset_for(spec.aug_type, nc)
            if col_offset is None or (nc.lineno, col_offset) not in mod_positions:  # type: ignore[attr-defined]
                continue
            for tracer in self._tracers:
                if spec in tracer.syntax_augmentation_specs():
                    tracer.augmented_node_ids_by_spec[spec].add(id(nc))
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
