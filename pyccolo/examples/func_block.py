# -*- coding: utf-8 -*-
"""
Statement-bodied blocks that ride the *subscript* path by passing a function.

:mod:`pyccolo.examples.brace_subscript` swaps ``macro{ ... }`` -> ``macro[ ... ]``
so existing subscript macros work with braces -- but only for *expression*
bodies, since a subscript slice must be an expression.

This example lifts that restriction for the *function-consuming* macro style. A
``{ ... }`` block may now contain full statements; the body is compiled into a
function and the **function** (an expression) is what gets passed to the
subscript::

    run{
        total = 0
        for i in range(5):
            total += i
        total
    }                       # parsed as  run[__pyc_fn__('<body>', globals(), locals())]
                            # -> run[<function>] -> the function is called -> 10

The trick is :attr:`AugmentationSpec.body_func_wrapper`: instead of splicing the
body into the slice verbatim, the paired augmenter wraps it as
``__pyc_fn__('<body source>', globals(), locals())`` -- a call expression that
evaluates to a freshly-``def``'d function. The macro then simply receives a
callable.

This does *not* retrofit statement bodies onto AST-template macros
(quick_lambda / pipescript), whose handlers consume ``node.slice`` as a code
template. It enables macros whose contract is "give me a function": here ``run``
(call it now, return the value) and ``thunk`` (return the function, deferred).
Both are plain objects with ``__getitem__`` -- but an ``after_subscript_slice``
pyccolo handler would receive the same function value just as well.

Closures are by value snapshot of ``globals()``/``locals()`` at the block site,
and nested blocks work because the wrapper re-runs the Pyccolo transform on the
captured body before compiling (so a ``run{...}`` inside another ``run{...}`` is
itself rewritten).
"""
import ast
import builtins
import textwrap
from typing import Any, Callable, Dict

import pyccolo as pyc
from pyccolo.examples.block_lambda import _build_function, _with_value_return

# names that open a function-bodied block
BLOCK_MACROS = ("run", "thunk")


class _CallBlock:
    """``run[fn]`` -> ``fn()`` (run the block now, yield its value)."""

    def __getitem__(self, fn: Callable[[], Any]) -> Any:
        return fn()


class _ThunkBlock:
    """``thunk[fn]`` -> ``fn`` (defer; hand back the callable)."""

    def __getitem__(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        return fn


class FuncBlockTracer(pyc.BaseTracer):
    global_guards_enabled = False

    block_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.subscript,
        token="{",
        replacement="[",
        close_token="}",
        close_replacement="]",
        name_pattern="|".join(BLOCK_MACROS),
        body_func_wrapper="__pyc_fn__",
    )

    def enter_tracing_hook(self) -> None:
        builtins.__pyc_fn__ = self.__pyc_fn__  # type: ignore[attr-defined]
        builtins.run = _CallBlock()  # type: ignore[attr-defined]
        builtins.thunk = _ThunkBlock()  # type: ignore[attr-defined]

    def exit_tracing_hook(self) -> None:
        for name in ("__pyc_fn__", "run", "thunk"):
            if hasattr(builtins, name):
                delattr(builtins, name)

    def __pyc_fn__(
        self, body_src: str, g: Dict[str, Any], l: Dict[str, Any]
    ) -> Callable[[], Any]:
        # re-transform so nested run{...}/thunk{...} blocks are handled
        body_src = self.transform(body_src)
        tree = ast.parse(textwrap.dedent(body_src))
        return _build_function(_with_value_return(tree.body), {**g, **l})
