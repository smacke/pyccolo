"""
Example of an quasiquoter for Pyccolo, similar to MacroPy's.
Ref: https://macropy3.readthedocs.io/en/latest/reference.html#quasiquote
"""
import ast
import pyccolo as pyc


class QuasiQuoter(pyc.BaseTracer):
    class _IdentitySubscript:
        def __getitem__(self, item):
            return item

    _identity_subscript = _IdentitySubscript()

    @pyc.init_module
    def init_module(self, *_, **__):
        import builtins

        # need to create a dummy reference to avoid NameError
        builtins.q = None

    def exit_tracing_hook(self) -> None:
        import builtins

        if hasattr(builtins, "q"):
            delattr(builtins, "q")

    @pyc.before_subscript_slice(
        when=lambda node: hasattr(node, "value")
        and isinstance(node.value, ast.Name)
        and node.value.id == "q"
    )
    def quote_slice_handler(self, _ret, node, *_, **__):
        if isinstance(node.slice, ast.Index):
            return lambda: node.slice.value
        else:
            return lambda: node.slice

    @pyc.load_name(when=lambda node: node.id == "q")
    def load_q(self, *_, **__):
        return self._identity_subscript
