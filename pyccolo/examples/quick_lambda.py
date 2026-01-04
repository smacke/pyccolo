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
import builtins
from functools import reduce
from types import FrameType
from typing import Any, Dict, List, Tuple, cast

import pyccolo as pyc
from pyccolo import fast
from pyccolo.examples.pipeline_tracer import PipelineTracer, SingletonArgCounterMixin
from pyccolo.examples.quasiquote import Quasiquoter, is_macro
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent


class _ArgReplacer(ast.NodeVisitor, SingletonArgCounterMixin):
    def __init__(self) -> None:
        super().__init__()
        self.placeholder_names: Dict[str, None] = {}

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id in QuickLambdaTracer.lambda_macros
        ):
            # defer visiting nested quick lambdas
            return
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            node.id != "_"
            and id(node)
            not in PipelineTracer.augmented_node_ids_by_spec[
                PipelineTracer.arg_placeholder_spec
            ]
        ):
            return
        # quick lambda will interpret this node as placeholder without any aug spec necessary
        PipelineTracer.augmented_node_ids_by_spec[
            PipelineTracer.arg_placeholder_spec
        ].discard(id(node))
        assert node.id.startswith("_")
        if node.id == "_":
            node.id = f"_{self.arg_ctr}"
            self.arg_ctr += 1
        else:
            if node.id[1].isalpha():
                node.id = node.id[1:]
            self.placeholder_names[node.id] = None

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.BitOr) and PipelineTracer.get_augmentations(
            id(node)
        ):
            return
        self.generic_visit(node)

    def get_placeholder_names(self, node: ast.AST) -> List[str]:
        self.placeholder_names.clear()
        self.visit(node)
        return list(self.placeholder_names)


class QuickLambdaTracer(Quasiquoter):

    global_guards_enabled = False

    lambda_macros = ("f", "filter", "ifilter", "map", "imap", "reduce")

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        for macro in self.lambda_macros:
            self.macros.add(macro)
        self._arg_replacer = _ArgReplacer()
        builtins.reduce = reduce  # type: ignore[attr-defined]
        builtins.imap = map  # type: ignore[attr-defined]
        self.lambda_cache: Dict[Tuple[int, int, TraceEvent], Any] = {}

    _not_found = object()

    @pyc.before_subscript_slice(when=is_macro(lambda_macros), reentrant=True)
    def handle_quick_lambda(
        self, _ret, node: ast.Subscript, frame: FrameType, evt: TraceEvent, *_, **__
    ):
        lambda_cache_key = (id(node), id(frame), evt)
        cached_lambda = self.lambda_cache.get(lambda_cache_key, self._not_found)
        if cached_lambda is not self._not_found:
            return cached_lambda
        __hide_pyccolo_frame__ = True
        orig_ctr = self._arg_replacer.arg_ctr
        orig_lambda_body: ast.expr = node.slice  # type: ignore[assignment]
        if isinstance(orig_lambda_body, getattr(ast, "Index", type(None))):
            orig_lambda_body = orig_lambda_body.value  # type: ignore[attr-defined]
        lambda_body = StatementMapper.bookkeeping_propagating_copy(orig_lambda_body)
        placeholder_names = self._arg_replacer.get_placeholder_names(lambda_body)
        if self._arg_replacer.arg_ctr == orig_ctr and len(placeholder_names) == 0:
            ast_lambda = lambda_body
        else:
            ast_lambda = SingletonArgCounterMixin.create_placeholder_lambda(
                placeholder_names, orig_ctr, lambda_body, frame.f_globals
            )
            ast_lambda.body = lambda_body
        func = cast(ast.Name, node.value).id
        if func in ("filter", "ifilter", "map", "imap", "reduce"):
            with fast.location_of(ast_lambda):
                arg = f"_{self._arg_replacer.arg_ctr}"
                self._arg_replacer.arg_ctr += 1
                inner_func = func
                if func == "ifilter":
                    inner_func = "filter"
                elif func == "imap":
                    inner_func = "map"
                lambda_body_str = f"{inner_func}(None, {arg})"
                functor_lambda_body = cast(
                    ast.Call,
                    cast(
                        ast.Expr,
                        fast.parse(lambda_body_str).body[0],
                    ).value,
                )
                functor_lambda_body.args[0] = ast_lambda
                if func in ("filter", "map"):
                    id_arg = f"_{self._arg_replacer.arg_ctr}"
                    self._arg_replacer.arg_ctr += 1
                    lambda_body_str = f"(list if type({arg}) is list else lambda {id_arg}: {id_arg})(None)"
                    functor_lambda_outer_body = cast(
                        ast.Call,
                        cast(
                            ast.Expr,
                            fast.parse(lambda_body_str).body[0],
                        ).value,
                    )
                    functor_lambda_outer_body.args[0] = functor_lambda_body
                    functor_lambda_body = functor_lambda_outer_body
                functor_lambda = cast(
                    ast.Lambda,
                    cast(ast.Expr, fast.parse(f"lambda {arg}: None").body[0]).value,
                )
                functor_lambda.body = functor_lambda_body
            ast_lambda = functor_lambda
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        ret = lambda: __hide_pyccolo_frame__ and evaluated_lambda  # noqa: E731
        self.lambda_cache[lambda_cache_key] = ret
        return ret
