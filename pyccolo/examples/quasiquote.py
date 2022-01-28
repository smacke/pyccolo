# -*- coding: utf-8 -*-
"""
Example of an quasiquoter for Pyccolo, similar to MacroPy's.
Ref: https://macropy3.readthedocs.io/en/latest/reference.html#quasiquote

Example:
```
a = 10
b = 2
q[1 + u[a + b]]  -> BinOp(Add, left=Num(1), right=Num(12))
```
"""
import ast
import copy
from typing import Callable

import pyccolo as pyc


class _QuasiquoteTransformer(ast.NodeTransformer):
    def __init__(self, global_env, local_env):
        self._global_env = global_env
        self._local_env = local_env

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(
            node.value, ast.Name
        ) and node.value.id in Quasiquoter.instance().macros - {"q"}:
            return pyc.eval(node, self._global_env, self._local_env)
        else:
            return node


def is_macro(name: str) -> Callable[[ast.AST], bool]:
    return (
        lambda node: isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == name
    )


class _IdentitySubscript:
    def __getitem__(self, item):
        return item


_identity_subscript = _IdentitySubscript()


class Quasiquoter(pyc.BaseTracer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.macros = {"q", "u", "name", "ast_literal", "ast_list"}

    def enter_tracing_hook(self) -> None:
        import builtins

        # need to create dummy reference to avoid NameError
        for macro in self.macros:
            if not hasattr(builtins, macro):
                setattr(builtins, macro, None)

    def exit_tracing_hook(self) -> None:
        import builtins

        for macro in self.macros:
            if hasattr(builtins, macro):
                delattr(builtins, macro)

    @pyc.before_subscript_slice(when=is_macro("q"), reentrant=True)
    def quote_handler(self, _ret, node, frame, *_, **__):
        to_visit = node.slice
        if isinstance(node.slice, ast.Index):
            to_visit = to_visit.value
        return lambda: _QuasiquoteTransformer(frame.f_globals, frame.f_locals).visit(
            copy.deepcopy(to_visit)
        )

    @pyc.after_subscript_slice(when=is_macro("u"), reentrant=True)
    def unquote_handler(self, ret, *_, **__):
        return pyc.eval(f"q[{repr(ret)}]")

    @pyc.after_subscript_slice(when=is_macro("name"), reentrant=True)
    def name_handler(self, ret, *_, **__):
        assert isinstance(ret, str)
        return pyc.eval(f"q[{ret}]")

    @pyc.after_subscript_slice(when=is_macro("ast_literal"), reentrant=True)
    def ast_literal_handler(self, ret, *_, **__):
        # technically we could get away without even having this handler
        assert isinstance(ret, ast.AST)
        return ret

    @pyc.after_subscript_slice(when=is_macro("ast_list"), reentrant=True)
    def ast_list_handler(self, ret, *_, **__):
        return ast.List(elts=list(ret))

    def is_any_macro(self, node):
        return any(is_macro(m)(node) for m in self.macros)

    @pyc.before_subscript_load(when=is_any_macro, reentrant=True)
    def load_macro_result(self, _ret, *_, **__):
        return _identity_subscript
