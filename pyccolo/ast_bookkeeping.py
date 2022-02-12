# -*- coding: utf-8 -*-
import ast
from typing import Dict, Optional


class BookkeepingVisitor(ast.NodeVisitor):
    def __init__(
        self,
        ast_node_by_id: Dict[int, ast.AST],
        containing_ast_by_id: Dict[int, ast.AST],
        containing_stmt_by_id: Dict[int, ast.stmt],
        parent_stmt_by_id: Dict[int, ast.stmt],
        stmt_by_lineno: Dict[int, ast.stmt],
    ):
        self.ast_node_by_id = ast_node_by_id
        self.containing_ast_by_id = containing_ast_by_id
        self.containing_stmt_by_id = containing_stmt_by_id
        self.parent_stmt_by_id = parent_stmt_by_id
        self.stmt_by_lineno = stmt_by_lineno
        self._current_containing_stmt: Optional[ast.stmt] = None

    def generic_visit(self, node: ast.AST):
        if isinstance(node, ast.stmt):
            self._current_containing_stmt = node
        self.ast_node_by_id[id(node)] = node
        if isinstance(node, ast.stmt):
            self.stmt_by_lineno[node.lineno] = node
            # workaround for python >= 3.8 wherein function calls seem
            # to yield trace frames that use the lineno of the first decorator
            for decorator in getattr(node, "decorator_list", []):
                self.stmt_by_lineno[decorator.lineno] = node
        for name, field in ast.iter_fields(node):
            if isinstance(field, ast.AST):
                self.containing_ast_by_id[id(field)] = node
                if self._current_containing_stmt is not None:
                    self.containing_stmt_by_id[
                        id(field)
                    ] = self._current_containing_stmt
            elif isinstance(field, list):
                for subfield in field:
                    if isinstance(subfield, ast.AST):
                        self.containing_ast_by_id[id(subfield)] = node
                        if isinstance(node, ast.stmt):
                            self.parent_stmt_by_id[id(subfield)] = node
        super().generic_visit(node)
