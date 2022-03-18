# -*- coding: utf-8 -*-
import ast
import builtins
import sys
from typing import (
    TYPE_CHECKING,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Union,
)

from pyccolo import fast
from pyccolo.extra_builtins import EMIT_EVENT, TRACE_LAMBDA
from pyccolo.trace_events import BEFORE_EXPR_EVENTS, TraceEvent

if TYPE_CHECKING:
    from pyccolo.ast_rewriter import GUARD_DATA_T
    from pyccolo.tracer import BaseTracer

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
    nullable_conditions: Iterable[Optional[Union[str, ast.expr]]],
    op: Optional[ast.AST] = None,
) -> ast.expr:
    conditions = [
        fast.Name(cond, ast.Load()) if isinstance(cond, str) else cond
        for cond in nullable_conditions
        if cond is not None
    ]
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
        tracers: "List[BaseTracer]",
        orig_to_copy_mapping: Dict[int, ast.AST],
        handler_predicate_by_event: DefaultDict[TraceEvent, Callable[..., bool]],
        handler_guards_by_event: DefaultDict[TraceEvent, List["GUARD_DATA_T"]],
    ):
        self.tracers = tracers
        self.orig_to_copy_mapping = orig_to_copy_mapping
        self.handler_predicate_by_event = handler_predicate_by_event
        self.handler_guards_by_event = handler_guards_by_event
        self.guards: Set[str] = tracers[-1].guards

    @staticmethod
    def is_tracing_disabled_context(node: ast.AST):
        if not isinstance(node, ast.With):
            return False
        if len(node.items) != 1:
            return False
        expr = node.items[0].context_expr
        if not isinstance(expr, ast.Call):
            return False
        func = expr.func
        if not isinstance(func, ast.Attribute):
            return False
        return (
            isinstance(func.value, ast.Name)
            and func.value.id == "pyc"
            and func.attr == "tracing_disabled"
        )

    def register_guard(self, guard: str) -> None:
        self.guards.add(guard)
        setattr(builtins, guard, True)

    @staticmethod
    def make_func_name(name=EMIT_EVENT) -> ast.Name:
        return fast.Name(name, ast.Load())

    def get_copy_node(self, orig_node_id: Union[int, ast.AST]) -> ast.AST:
        if not isinstance(orig_node_id, int):
            orig_node_id = id(orig_node_id)
        return self.orig_to_copy_mapping[orig_node_id]

    def get_copy_id_ast(self, orig_node_id: Union[int, ast.AST]) -> NumConst:
        return fast.Num(id(self.get_copy_node(orig_node_id)))

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
    ) -> Union[ast.Call, ast.IfExp]:
        args = args or []
        before_expr_args = before_expr_args or []
        if evt in BEFORE_EXPR_EVENTS and "ret" in kwargs:
            kwargs_ret = kwargs["ret"]
            if (
                not isinstance(kwargs_ret, ast.Call)
                or not isinstance(kwargs_ret.func, ast.Name)
                or kwargs_ret.func.id != TRACE_LAMBDA
            ):
                kwargs["ret"] = self.make_lambda(kwargs_ret)
        local_guard_makers = self.handler_guards_by_event.get(evt, None)
        local_guards = {}
        if local_guard_makers is not None:
            for spec, maker in local_guard_makers:
                guardval = maker(node_or_id)
                if guardval is not None:
                    local_guards[id(spec)] = guardval
        if len(local_guards) == 0:
            kwargs["guards_by_handler_spec_id"] = fast.NameConstant(None)
        else:
            kwargs["guards_by_handler_spec_id"] = fast.Dict(
                keys=[fast.Num(k) for k in local_guards.keys()],
                values=[fast.Str(v) for v in local_guards.values()],
            )
        ret: Union[ast.Call, ast.IfExp] = fast.Call(
            func=self.make_func_name(),
            args=[evt.to_ast(), self.get_copy_id_ast(node_or_id)] + args,
            keywords=fast.kwargs(**kwargs),
        )
        if evt in BEFORE_EXPR_EVENTS:
            ret = fast.Call(func=ret, args=before_expr_args)
        if len(local_guards) > 0:
            for guard in local_guards.values():
                self.tracers[-1].register_local_guard(guard)
            ret = fast.IfExp(
                test=make_composite_condition(local_guards.values(), op=fast.Or()),
                body=self.get_copy_node(node_or_id),
                orelse=ret,
            )
        return ret
