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


class ContainingStatementMapper(ast.NodeVisitor):
    def __init__(
        self, node_id_to_stmt_map: Dict[int, ast.stmt], containing_stmt: ast.stmt
    ):
        self.node_id_to_stmt_map: Dict[int, ast.stmt] = node_id_to_stmt_map
        self.containing_stmt: ast.stmt = containing_stmt

    def generic_visit(self, node: ast.AST):
        self.node_id_to_stmt_map[id(node)] = self.containing_stmt
        super().generic_visit(node)


class StatementMapper(ast.NodeVisitor):
    def __init__(
        self,
        line_to_stmt_map: Dict[int, ast.stmt],
        tracers: "List[BaseTracer]",
        augmented_positions_by_spec: Dict[AugmentationSpec, Set[Tuple[int, int]]],
    ):
        self.line_to_stmt_map: Dict[int, ast.stmt] = line_to_stmt_map
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
            return getattr(node.value, "end_col_offset", -1)
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

    def __call__(
        self, node: Union[ast.Module, ast.FunctionDef, ast.AsyncFunctionDef]
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
            self._tracers[-1].ast_node_by_id[id(nc)] = nc
            if hasattr(nc, "lineno"):
                for spec, mod_positions in self.augmented_positions_by_spec.items():
                    col_offset = self._get_col_offset_for(spec.aug_type, nc)
                    if col_offset is None:
                        continue
                    if (nc.lineno, col_offset) in mod_positions:
                        for tracer in self._tracers:
                            if spec in tracer.syntax_augmentation_specs:
                                tracer.augmented_node_ids_by_spec[spec].add(id(nc))
            if isinstance(nc, ast.stmt):
                self.line_to_stmt_map[nc.lineno] = nc
                ContainingStatementMapper(
                    self._tracers[-1].node_id_to_containing_stmt, nc
                ).visit(nc)
                # workaround for python >= 3.8 wherein function calls seem
                # to yield trace frames that use the lineno of the first decorator
                for decorator in getattr(nc, "decorator_list", []):
                    self.line_to_stmt_map[decorator.lineno] = nc
                nc_body = getattr(nc, "body", [])
                try:
                    for child in nc_body:
                        if isinstance(child, ast.AST):
                            self._tracers[-1].parent_node_by_id[id(child)] = nc
                except TypeError:
                    self._tracers[-1].parent_node_by_id[id(nc_body)] = nc
                for name, field in ast.iter_fields(nc):
                    if name == "body":
                        continue
                    if isinstance(field, list):
                        for child in field:
                            if isinstance(child, ast.AST):
                                self._tracers[-1].parent_node_by_id[id(child)] = nc
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
