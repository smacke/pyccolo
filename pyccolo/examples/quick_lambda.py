# -*- coding: utf-8 -*-
"""
Implementation of quick lambdas in Pyccolo, similar to MacroPy's.
Ref: https://macropy3.readthedocs.io/en/latest/quick_lambda.html#quicklambda

Example:
```
with QuickLambdaTracer:
    pyc.eval("f[_ + _](3, 4)")
>>> 7
```
"""
import ast
from typing import cast

import pyccolo as pyc
from pyccolo.examples.quasiquote import Quasiquoter, is_macro
from pyccolo.stmt_mapper import StatementMapper


class _ArgReplacer(ast.NodeVisitor):
    def __init__(self):
        self.ctr = 0

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "f":
            # defer visiting nested quick lambdas
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == "_":
            node.id = f"_{self.ctr}"
            self.ctr += 1


class QuickLambdaTracer(Quasiquoter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macros.add("f")
        self._arg_replacer = _ArgReplacer()

    @pyc.before_subscript_slice(when=is_macro("f"), reentrant=True)
    def handle_quick_lambda(self, _ret, node, frame, *_, **__):
        orig_ctr = self._arg_replacer.ctr
        orig_lambda_body = node.slice
        if isinstance(node.slice, ast.Index):
            orig_lambda_body = orig_lambda_body.value
        lambda_body = StatementMapper.augmentation_propagating_copy(orig_lambda_body)
        self._arg_replacer.visit(lambda_body)
        num_lambda_args = self._arg_replacer.ctr - orig_ctr
        lambda_args = ", ".join(
            f"_{arg_idx}" for arg_idx in range(orig_ctr, orig_ctr + num_lambda_args)
        )
        ast_lambda = cast(
            ast.Expr, ast.parse(f"lambda {lambda_args}: None").body[0]
        ).value
        ast_lambda.body = lambda_body
        return lambda: pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
