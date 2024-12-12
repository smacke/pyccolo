# -*- coding: utf-8 -*-
import ast
from typing import Dict, NamedTuple, Optional, Tuple


class AstBookkeeper(NamedTuple):
    path: str
    module_id: int
    ast_node_by_id: Dict[int, ast.AST]
    containing_ast_by_id: Dict[int, ast.AST]
    containing_stmt_by_id: Dict[int, ast.stmt]
    parent_stmt_by_id: Dict[int, ast.stmt]
    stmt_by_lineno: Dict[int, ast.stmt]

    def remap(self, new_module_id: int) -> Tuple["AstBookkeeper", Dict[int, int]]:
        """
        After unpickling, the ast nodes will have different ids than before.
        This method will compuate a new bookkeeper to reflect the new ids, as well
        as return a mapping from the old ids to the new ids.
        """
        remapping: Dict[int, int] = {self.module_id: new_module_id}
        new_ast_node_by_id: Dict[int, ast.AST] = {}
        new_containing_ast_by_id: Dict[int, ast.AST] = {}
        new_containing_stmt_by_id: Dict[int, ast.stmt] = {}
        new_parent_stmt_by_id: Dict[int, ast.stmt] = {}
        for old_id, ast_node in self.ast_node_by_id.items():
            new_id = id(ast_node)
            remapping[old_id] = new_id
            new_ast_node_by_id[new_id] = ast_node
            if old_id in self.containing_ast_by_id:
                new_containing_ast_by_id[new_id] = self.containing_ast_by_id[old_id]
            if old_id in self.containing_stmt_by_id:
                new_containing_stmt_by_id[new_id] = self.containing_stmt_by_id[old_id]
            if old_id in self.parent_stmt_by_id:
                new_parent_stmt_by_id[new_id] = self.parent_stmt_by_id[old_id]
        return (
            AstBookkeeper(
                self.path,
                new_module_id,
                new_ast_node_by_id,
                new_containing_ast_by_id,
                new_containing_stmt_by_id,
                new_parent_stmt_by_id,
                self.stmt_by_lineno,
            ),
            remapping,
        )

    @classmethod
    def create(cls, path: str, module_id: int) -> "AstBookkeeper":
        return cls(path, module_id, {}, {}, {}, {}, {})


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
        if self._current_containing_stmt is not None:
            self.containing_stmt_by_id.setdefault(
                id(node), self._current_containing_stmt
            )
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
