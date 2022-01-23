"""
Example of an quasiquoter for Pyccolo, similar to MacroPy's.
Ref: https://macropy3.readthedocs.io/en/latest/reference.html#quasiquote

Currently 'q', 'u', and 'name' operators are supported; e.g.,
a = 10
b = 2
q[1 + u[a + b]]  -> BinOp(Add, left=Num(1), right=Num(12))
"""
import ast
import copy

import pyccolo as pyc


_MACROS = {"q", "u", "name"}


class _ReplaceUnquoteTransformer(ast.NodeTransformer):
    def __init__(self, local_env, global_env):
        self._local_env = local_env
        self._global_env = global_env

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in _MACROS - {"q"}:
            to_wrap = node.slice
            if isinstance(to_wrap, ast.Index):
                to_wrap = to_wrap.value  # type: ignore
            raw = pyc.eval(to_wrap, self._local_env, self._global_env)
            if node.value.id == "name":
                return ast.Name(raw, ast.Load())  # type: ignore
            else:
                return ast.parse(repr(raw)).body[0].value  # type: ignore
        else:
            return node


def is_subscript(name):
    return (
        lambda node: hasattr(node, "value")
        and isinstance(node.value, ast.Name)
        and node.value.id == name
    )


class QuasiQuoter(pyc.BaseTracer):
    class _IdentitySubscript:
        def __getitem__(self, item):
            return item

    _identity_subscript = _IdentitySubscript()

    def enter_tracing_hook(self) -> None:
        import builtins

        # need to create dummy reference to avoid NameError
        for macro in _MACROS:
            if not hasattr(builtins, macro):
                setattr(builtins, macro, None)

    def exit_tracing_hook(self) -> None:
        import builtins

        for macro in _MACROS:
            if hasattr(builtins, macro):
                delattr(builtins, macro)

    @pyc.before_subscript_slice(when=is_subscript("q"))
    def quote_handler(self, _ret, node, frame, *_, **__):
        to_visit = node.slice
        if isinstance(node.slice, ast.Index):
            to_visit = to_visit.value
        return lambda: _ReplaceUnquoteTransformer(
            frame.f_locals, frame.f_globals
        ).visit(copy.deepcopy(to_visit))

    @pyc.before_subscript_load(when=is_subscript("q"))
    def load_q(self, *_, **__):
        return self._identity_subscript
