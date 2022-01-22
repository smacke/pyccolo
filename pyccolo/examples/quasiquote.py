"""
Example of an quasiquoter for Pyccolo, similar to MacroPy's.
Ref: https://macropy3.readthedocs.io/en/latest/reference.html#quasiquote
"""
import ast

import pyccolo as pyc


class _ReplaceUnquoteTransformer(ast.NodeTransformer):
    def __init__(self, local_env, global_env):
        self._local_env = local_env
        self._global_env = global_env

    def visit_Subscript(self, node: ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id == "u":
            to_wrap = node.slice
            if isinstance(to_wrap, ast.Index):
                to_wrap = to_wrap.value
            return (
                ast.parse(
                    repr(
                        eval(
                            compile(ast.Expression(to_wrap), "<file>", "eval"),
                            self._global_env,
                            self._local_env,
                        )
                    )
                )
                .body[0]
                .value
            )
        else:
            return node


class QuasiQuoter(pyc.BaseTracer):
    class _IdentitySubscript:
        def __getitem__(self, item):
            return item

    _identity_subscript = _IdentitySubscript()

    @pyc.init_module
    def init_module(self, *_, **__):
        import builtins

        # need to create dummy references to avoid NameError
        builtins.q = None
        builtins.u = None

    def exit_tracing_hook(self) -> None:
        import builtins

        if hasattr(builtins, "q"):
            delattr(builtins, "q")
        if hasattr(builtins, "u"):
            delattr(builtins, "u")

    @pyc.before_subscript_slice(
        when=lambda node: hasattr(node, "value")
        and isinstance(node.value, ast.Name)
        and node.value.id == "q"
    )
    def quote_slice_handler(self, _ret, node, frame, *_, **__):
        to_visit = node.slice
        if isinstance(node.slice, ast.Index):
            to_visit = to_visit.value
        return lambda: _ReplaceUnquoteTransformer(
            frame.f_locals, frame.f_globals
        ).visit(to_visit)

    @pyc.load_name(when=lambda node: node.id == "q")
    def load_q(self, *_, **__):
        return self._identity_subscript
