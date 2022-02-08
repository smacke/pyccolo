# -*- coding: utf-8 -*-
import ast
import copy
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

from pyccolo.syntax_augmentation import AugmentationSpec, AugmentationType

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class StatementMapper(ast.NodeVisitor):
    def __init__(
        self,
        tracers: "List[BaseTracer]",
        augmented_positions_by_spec: Dict[AugmentationSpec, Set[Tuple[int, int]]],
    ):
        self._tracers: "List[BaseTracer]" = tracers
        self.augmented_positions_by_spec = augmented_positions_by_spec
        self.traversal: List[ast.AST] = []

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
        else:
            return None

    @staticmethod
    def _get_dot_col_offset_for(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Attribute):
            return getattr(node.value, "end_col_offset", -1)
        else:
            return None

    @staticmethod
    def _get_binop_col_offset_for(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.BinOp):
            return getattr(node.left, "end_col_offset", -2) + 1
        else:
            return None

    def _get_col_offset_for(
        self, aug_type: AugmentationType, node: ast.AST
    ) -> Optional[int]:
        if aug_type == AugmentationType.prefix:
            return self._get_prefix_col_offset_for(node)
        elif aug_type == AugmentationType.suffix:
            return self._get_suffix_col_offset_for(node)
        elif aug_type == AugmentationType.dot:
            return self._get_dot_col_offset_for(node)
        elif aug_type == AugmentationType.binop:
            return self._get_binop_col_offset_for(node)
        else:
            raise NotImplementedError()

    def _handle_augmentations(self, nc: ast.AST) -> None:
        for spec, mod_positions in self.augmented_positions_by_spec.items():
            col_offset = self._get_col_offset_for(spec.aug_type, nc)
            if col_offset is None:
                continue
            if (nc.lineno, col_offset) in mod_positions:
                for tracer in self._tracers:
                    if spec in tracer.syntax_augmentation_specs:
                        tracer.augmented_node_ids_by_spec[spec].add(id(nc))

    def __call__(
        self,
        node: Union[ast.Expression, ast.Module, ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Dict[int, ast.AST]:
        # for some bizarre reason we need to visit once to clear empty nodes apparently
        self.visit(node)
        self.traversal.clear()

        self.visit(node)
        orig_traversal = self.traversal
        self.traversal = []
        self.visit(copy.deepcopy(node))
        copy_traversal = self.traversal
        orig_to_copy_mapping = {}
        for no, nc in zip(orig_traversal, copy_traversal):
            orig_to_copy_mapping[id(no)] = nc
            if hasattr(nc, "lineno"):
                self._handle_augmentations(nc)
        return orig_to_copy_mapping

    def visit(self, node):
        self.traversal.append(node)
        for name, field in ast.iter_fields(node):
            if isinstance(field, ast.AST):
                self.visit(field)
            elif isinstance(field, list):
                for inner_node in field:
                    if isinstance(inner_node, ast.AST):
                        self.visit(inner_node)
