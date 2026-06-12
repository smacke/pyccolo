# -*- coding: utf-8 -*-
"""
Statement-bodied "multi-line lambda" blocks for Pyccolo.

Where :mod:`pyccolo.examples.quick_lambda` gives you *expression*-bodied quick
lambdas via the already-valid subscript syntax (``map[_ + 1]``), this example
demonstrates correlating a pair of delimiters so that a whole *statement* body
can be captured and turned into a function at runtime::

    with BlockLambdaTracer:
        # a statement-bodied generator expression
        assert pyc.eval('''list(map{
            for i in range(10):
                if i % 2 == 0:
                    yield i * i
        })''') == [0, 4, 16, 36, 64]

        # an immediately-invoked multi-line lambda (value of the last expr)
        assert pyc.eval('''do{
            total = 0
            for i in range(5):
                total += i
            total
        }''') == 10

        # a deferred zero-arg function
        with BlockLambdaTracer:
            f = pyc.eval('''fn{
                xs = []
                for i in range(3):
                    xs.append(i)
                xs
            }''')
            assert f() == [0, 1, 2]

The brace pair is correlated by
:func:`pyccolo.syntax_augmentation.make_paired_delimiter_augmenter`, which
captures the raw source between ``trigger{`` and the matching ``}`` and rewrites
the span into a call to a runtime helper, ``__pyc_block__(trigger, body, ...)``.
The helper compiles ``body`` into a function in the call-site's namespace and
applies trigger-specific semantics.

Nesting works because the helper re-runs the Pyccolo transform on the captured
body before compiling, so nested ``trigger{ ... }`` blocks are rewritten too.
Closures are by *value snapshot*: the body reads enclosing locals/globals as they
were at the moment the block executed (assignments inside the block do not leak
back out).
"""
import ast
import builtins
import textwrap
from typing import Any, Callable, Dict, List

import pyccolo as pyc
from pyccolo.syntax_augmentation import make_paired_delimiter_augmenter


def _build_function(body: List[ast.stmt], namespace: Dict[str, Any]) -> Callable:
    """Compile ``body`` (a list of statements) into a zero-arg function defined
    in ``namespace`` and return the resulting callable."""
    template = ast.parse("def __pyc_block_fn__():\n    pass")
    func_def = template.body[0]
    func_def.body = body or [ast.Pass()]  # type: ignore[attr-defined]
    ast.fix_missing_locations(template)
    exec(compile(template, "<pyc-block>", "exec"), namespace)
    return namespace["__pyc_block_fn__"]


def _with_value_return(body: List[ast.stmt]) -> List[ast.stmt]:
    """If the block ends in a bare expression, turn it into a ``return`` so the
    compiled function yields that value."""
    if body and isinstance(body[-1], ast.Expr):
        last = body[-1]
        body = list(body[:-1]) + [ast.Return(value=last.value)]
    return body


class BlockLambdaTracer(pyc.BaseTracer):

    # this tracer only rewrites source; it doesn't need guard machinery.
    global_guards_enabled = False

    # names that, when written as ``name{ ... }``, open a statement block.
    block_macros = ("map", "gen", "do", "fn")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._paired_augment = make_paired_delimiter_augmenter(
            self.block_macros, self._emit
        )

    @staticmethod
    def _emit(trigger: str, body: str) -> str:
        return "__pyc_block__({!r}, {!r}, globals(), locals())".format(trigger, body)

    # --- wiring the paired augmenter into both the eval/exec and import paths ---

    def preprocess(self, code, rewriter):
        # eval / exec / transform path. super().preprocess applies any
        # token-based AugmentationSpecs (none here) and otherwise returns code.
        code = super().preprocess(code, rewriter)
        return self._paired_augment(code)

    def make_syntax_augmenter(self, ast_rewriter):
        # import path: TraceLoader applies the augmenter returned here.
        base = super().make_syntax_augmenter(ast_rewriter)

        def _aug(lines):
            out = base(lines)
            if isinstance(out, list):
                return self._paired_augment("".join(out)).splitlines(keepends=True)
            return self._paired_augment(out)

        return _aug

    # --- make the runtime helper resolvable while tracing ---

    def enter_tracing_hook(self) -> None:
        builtins.__pyc_block__ = self.__pyc_block__  # type: ignore[attr-defined]

    def exit_tracing_hook(self) -> None:
        if hasattr(builtins, "__pyc_block__"):
            delattr(builtins, "__pyc_block__")

    # --- the runtime helper ---

    def __pyc_block__(
        self, trigger: str, body_src: str, g: Dict[str, Any], loc: Dict[str, Any]
    ) -> Any:
        # re-transform so nested trigger{...} blocks inside the body are handled.
        body_src = self.transform(body_src)
        tree = ast.parse(textwrap.dedent(body_src))
        namespace: Dict[str, Any] = {**g, **loc}

        if trigger in ("map", "gen"):
            # statement-bodied generator expression: body is expected to contain
            # `yield`, making __pyc_block_fn__ a generator function; we call it
            # and hand back the resulting generator.
            func = _build_function(tree.body, namespace)
            return func()
        elif trigger == "do":
            # immediately-invoked multi-line lambda: returns the last expression.
            func = _build_function(_with_value_return(tree.body), namespace)
            return func()
        else:  # "fn" and any future deferred-callable triggers
            return _build_function(_with_value_return(tree.body), namespace)
