# -*- coding: utf-8 -*-
"""
Pyccolo: declarative, composable, portable instrumentation embedded directly in Python source.

Pyccolo brings metaprogramming to everybody via general event-emitting AST transformations.
"""
import ast
import functools
import inspect
import textwrap
import types
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Union

from pyccolo.ast_rewriter import AstRewriter
from pyccolo.emit_event import _TRACER_STACK, SkipAll, allow_reentrant_event_handling
from pyccolo.extra_builtins import PYCCOLO_BUILTIN_PREFIX, make_guard_name
from pyccolo.predicate import Predicate
from pyccolo.syntax_augmentation import AugmentationSpec, AugmentationType
from pyccolo.trace_events import TraceEvent
from pyccolo.trace_stack import TraceStack
from pyccolo.tracer import (
    BaseTracer,
    NoopTracer,
    Null,
    Pass,
    Skip,
    register_handler,
    register_raw_handler,
    skip_when_tracing_disabled,
)
from pyccolo.utils import multi_context, resolve_tracer

event = TraceEvent


call = TraceEvent.call
return_ = TraceEvent.return_
exception = TraceEvent.exception
before_import = TraceEvent.before_import
init_module = TraceEvent.init_module
exit_module = TraceEvent.exit_module
after_import = TraceEvent.after_import
before_stmt = TraceEvent.before_stmt
after_stmt = TraceEvent.after_stmt
after_module_stmt = TraceEvent.after_module_stmt
after_expr_stmt = TraceEvent.after_expr_stmt
load_name = TraceEvent.load_name
before_for_loop_body = TraceEvent.before_for_loop_body
after_for_loop_iter = TraceEvent.after_for_loop_iter
before_while_loop_body = TraceEvent.before_while_loop_body
after_while_loop_iter = TraceEvent.after_while_loop_iter
before_attribute_load = TraceEvent.before_attribute_load
before_attribute_store = TraceEvent.before_attribute_store
before_attribute_del = TraceEvent.before_attribute_del
after_attribute_load = TraceEvent.after_attribute_load
before_subscript_load = TraceEvent.before_subscript_load
before_subscript_store = TraceEvent.before_subscript_store
before_subscript_del = TraceEvent.before_subscript_del
after_subscript_load = TraceEvent.after_subscript_load
before_subscript_slice = TraceEvent.before_subscript_slice
after_subscript_slice = TraceEvent.after_subscript_slice
_load_saved_slice = TraceEvent._load_saved_slice
before_load_complex_symbol = TraceEvent.before_load_complex_symbol
after_load_complex_symbol = TraceEvent.after_load_complex_symbol
after_if_test = TraceEvent.after_if_test
after_while_test = TraceEvent.after_while_test
before_lambda = TraceEvent.before_lambda
after_lambda = TraceEvent.after_lambda
decorator = TraceEvent.decorator
before_call = TraceEvent.before_call
after_call = TraceEvent.after_call
before_argument = TraceEvent.before_argument
after_argument = TraceEvent.after_argument
before_return = TraceEvent.before_return
after_return = TraceEvent.after_return
before_dict_literal = TraceEvent.before_dict_literal
after_dict_literal = TraceEvent.after_dict_literal
before_list_literal = TraceEvent.before_list_literal
after_list_literal = TraceEvent.after_list_literal
before_set_literal = TraceEvent.before_set_literal
after_set_literal = TraceEvent.after_set_literal
before_tuple_literal = TraceEvent.before_tuple_literal
after_tuple_literal = TraceEvent.after_tuple_literal
dict_key = TraceEvent.dict_key
dict_value = TraceEvent.dict_value
list_elt = TraceEvent.list_elt
set_elt = TraceEvent.set_elt
tuple_elt = TraceEvent.tuple_elt
before_assign_rhs = TraceEvent.before_assign_rhs
after_assign_rhs = TraceEvent.after_assign_rhs
before_augassign_rhs = TraceEvent.before_augassign_rhs
after_augassign_rhs = TraceEvent.after_augassign_rhs
before_function_body = TraceEvent.before_function_body
after_function_execution = TraceEvent.after_function_execution
before_lambda_body = TraceEvent.before_lambda_body
after_lambda_body = TraceEvent.after_lambda_body
left_binop_arg = TraceEvent.left_binop_arg
right_binop_arg = TraceEvent.right_binop_arg
before_binop = TraceEvent.before_binop
after_binop = TraceEvent.after_binop
left_compare_arg = TraceEvent.left_compare_arg
compare_arg = TraceEvent.compare_arg
before_compare = TraceEvent.before_compare
after_compare = TraceEvent.after_compare
after_comprehension_if = TraceEvent.after_comprehension_if
after_comprehension_elt = TraceEvent.after_comprehension_elt
after_dict_comprehension_key = TraceEvent.after_dict_comprehension_key
after_dict_comprehension_value = TraceEvent.after_dict_comprehension_value
exception_handler_type = TraceEvent.exception_handler_type
ellipses = TraceEvent.ellipses


# redundant; do this just in case we forgot to add stubs in trace_events.py
for evt in TraceEvent:
    globals()[evt.name] = evt


# convenience functions for managing tracer singleton
def tracer() -> BaseTracer:
    if len(_TRACER_STACK) > 0:
        return _TRACER_STACK[-1]
    else:
        return NoopTracer.instance()


def instance() -> BaseTracer:
    return tracer()


def parse(code: str, mode: str = "exec") -> Union[ast.Module, ast.Expression]:
    return tracer().parse(code, mode=mode)


def eval(code: Union[str, ast.expr, ast.Expression], *args, **kwargs) -> Any:
    return tracer().eval(
        code,
        *args,
        num_extra_lookback_frames=kwargs.pop("num_extra_lookback_frames", 0) + 1,
        **kwargs,
    )


def exec(code: Union[str, ast.Module, ast.stmt], *args, **kwargs) -> Dict[str, Any]:
    return tracer().exec(
        code,
        *args,
        num_extra_lookback_frames=kwargs.pop("num_extra_lookback_frames", 0) + 1,
        **kwargs,
    )


def execute(*args, **kwargs) -> Dict[str, Any]:
    return exec(*args, **kwargs)


def instrumented(tracers: List[BaseTracer]) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        f_defined_file = f.__code__.co_filename
        with multi_context([tracer.tracing_disabled() for tracer in tracers]):
            code = ast.parse(textwrap.dedent(inspect.getsource(f)))
            code.body[0] = tracers[-1].make_ast_rewriter().visit(code.body[0])
            compiled: types.CodeType = compile(code, f.__code__.co_filename, "exec")
            for const in compiled.co_consts:
                if (
                    isinstance(const, types.CodeType)
                    and const.co_name == f.__code__.co_name
                ):
                    f.__code__ = const
                    break

        @functools.wraps(f)
        def instrumented_f(*args, **kwargs) -> Any:
            with multi_context(
                [
                    tracer.tracing_enabled(tracing_enabled_file=f_defined_file)
                    for tracer in tracers
                ]
            ):
                return f(*args, **kwargs)

        return instrumented_f

    return decorator


@contextmanager
def tracing_context(tracers=None, *args, **kwargs):
    tracers = _TRACER_STACK if tracers is None else tracers
    with multi_context([tracer.tracing_context(*args, **kwargs) for tracer in tracers]):
        yield


@contextmanager
def tracing_enabled(tracers=None, **kwargs):
    tracers = _TRACER_STACK if tracers is None else tracers
    if len(tracers) == 0:
        raise ValueError("Expected at least one tracer to enable")
    with multi_context([tracer.tracing_enabled(**kwargs) for tracer in tracers]):
        yield


@contextmanager
def tracing_disabled(tracers=None, **kwargs):
    if tracers is None:
        tracers = _TRACER_STACK
        if len(tracers) == 0:
            tracers = [NoopTracer.instance()]
    with multi_context([tracer.tracing_disabled(**kwargs) for tracer in tracers]):
        yield


is_outer_stmt = BaseTracer.is_outer_stmt


__all__ = [
    "__version__",
    "AstRewriter",
    "AugmentationSpec",
    "AugmentationType",
    "BaseTracer",
    "NoopTracer",
    "Null",
    "PYCCOLO_BUILTIN_PREFIX",
    "Pass",
    "Predicate",
    "Skip",
    "SkipAll",
    "TraceStack",
    "allow_reentrant_event_handling",
    "event",
    "exec",
    "execute",
    "instance",
    "instrumented",
    "is_outer_stmt",
    "make_guard_name",
    "multi_context",
    "parse",
    "register_handler",
    "register_raw_handler",
    "resolve_tracer",
    "skip_when_tracing_disabled",
    "tracer",
    "tracing_context",
    "tracing_disabled",
    "tracing_enabled",
]


# all the events now
__all__.extend(evt.name for evt in TraceEvent)

from pyccolo import _version  # noqa: E402
__version__ = _version.get_versions()['version']
