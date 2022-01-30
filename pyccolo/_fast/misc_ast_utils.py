# -*- coding: utf-8 -*-
import ast
import builtins
import sys
import typing
from typing import Callable, DefaultDict, Dict, List, Optional, Set, Union

from pyccolo import fast
from pyccolo.extra_builtins import EMIT_EVENT, TRACE_LAMBDA
from pyccolo.trace_events import BEFORE_EXPR_EVENTS, TraceEvent

if sys.version_info < (3, 8):
    NumConst = ast.Num
else:
    NumConst = ast.Constant


def make_test(var_name: str, negate: bool = False) -> ast.expr:
    ret: ast.expr = fast.Name(var_name, ast.Load())
    if negate:
        ret = fast.UnaryOp(operand=ret, op=fast.Not())
    return ret


def make_composite_condition(
    nullable_conditions: typing.List[Optional[ast.expr]], op: Optional[ast.AST] = None
):
    conditions = [cond for cond in nullable_conditions if cond is not None]
    if len(conditions) == 1:
        return conditions[0]
    op = op or fast.And()  # type: ignore
    return fast.BoolOp(op=op, values=conditions)


def subscript_to_slice(node: ast.Subscript) -> ast.expr:
    if isinstance(node.slice, ast.Index):
        return node.slice.value  # type: ignore
    else:
        return node.slice  # type: ignore


class EmitterMixin:
    def __init__(
        self,
        orig_to_copy_mapping: Dict[int, ast.AST],
        handler_predicate_by_event: DefaultDict[TraceEvent, Callable[..., bool]],
        guards: Set[str],
    ):
        self.orig_to_copy_mapping = orig_to_copy_mapping
        self.handler_predicate_by_event = handler_predicate_by_event
        self.guards: Set[str] = guards

    def register_guard(self, guard: str) -> None:
        self.guards.add(guard)
        setattr(builtins, guard, True)

    @staticmethod
    def make_func_name(name=EMIT_EVENT) -> ast.Name:
        return fast.Name(name, ast.Load())

    def get_copy_id_ast(self, orig_node_id: Union[int, ast.AST]) -> NumConst:
        if not isinstance(orig_node_id, int):
            orig_node_id = id(orig_node_id)
        return fast.Num(id(self.orig_to_copy_mapping[orig_node_id]))

    def make_lambda(
        self, body: ast.expr, args: Optional[List[ast.arg]] = None
    ) -> ast.Call:
        return fast.Call(
            func=self.make_func_name(TRACE_LAMBDA),
            args=[
                fast.Lambda(
                    body=body,
                    args=ast.arguments(
                        args=[] if args is None else args,
                        defaults=[],
                        kwonlyargs=[],
                        kw_defaults=[],
                        posonlyargs=[],
                    ),
                )
            ],
        )

    def emit(
        self,
        evt: TraceEvent,
        node_or_id: Union[int, ast.AST],
        args=None,
        before_expr_args=None,
        **kwargs,
    ) -> ast.Call:
        args = args or []
        before_expr_args = before_expr_args or []
        ret = fast.Call(
            func=self.make_func_name(),
            args=[evt.to_ast(), self.get_copy_id_ast(node_or_id)] + args,
            keywords=fast.kwargs(**kwargs),
        )
        if evt in BEFORE_EXPR_EVENTS:
            ret = fast.Call(func=ret, args=before_expr_args)
        return ret

    def make_tuple_event_for(
        self, node: ast.AST, event: TraceEvent, orig_node_id=None, **kwargs
    ):
        if not self.handler_predicate_by_event[event](node):
            return node
        with fast.location_of(node):
            tuple_node = fast.Tuple(
                [self.emit(event, orig_node_id or node, **kwargs), node], ast.Load()
            )
            slc: Union[ast.Constant, ast.Num, ast.Index] = fast.Num(1)
            if sys.version_info < (3, 9):
                slc = fast.Index(slc)
            return fast.Subscript(tuple_node, slc, ast.Load())
