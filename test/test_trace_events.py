# -*- coding: utf-8 -*-
import difflib
import functools
import logging
from types import FrameType
from typing import List, Set, Union

import hypothesis.strategies as st
from hypothesis import example, given, settings

import pyccolo as pyc
from pyccolo.trace_events import TraceEvent

logging.basicConfig(level=logging.INFO)


_RECORDED_EVENTS = []
_DIFFER = difflib.Differ()
_FILENAME = "<trace_event_sandbox>"


def patch_events_and_emitter(testfunc):
    @functools.wraps(testfunc)
    @settings(max_examples=25, deadline=None)
    @example(events=set(pyc.TraceEvent))
    def wrapped_testfunc(events):
        if events & {
            pyc.before_subscript_load,
            pyc.before_subscript_store,
            pyc.before_subscript_del,
        }:
            events.add(TraceEvent._load_saved_slice)
        if TraceEvent._load_saved_slice in events:
            events.add(pyc.before_subscript_load)
            events.add(pyc.before_subscript_store)
            events.add(pyc.before_subscript_del)
        if events & {
            TraceEvent._load_saved_expr_stmt_ret,
            pyc.after_module_stmt,
        }:
            events.add(TraceEvent._load_saved_expr_stmt_ret)
            events.add(pyc.after_stmt)
            events.add(pyc.after_module_stmt)
        # get rid of all the builtin tracing events since we'll have all if we have one
        events -= {
            pyc.line,
            pyc.call,
            pyc.c_call,
            pyc.return_,
            pyc.c_return,
            pyc.exception,
            pyc.c_exception,
            pyc.opcode,
        }

        orig_init = pyc.BaseTracer.__init__
        original_emit_event = pyc.BaseTracer._emit_event

        def patched_init(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self.events_with_registered_handlers = frozenset(events)

        def _patched_emit_event(
            self,
            event: Union[str, pyc.TraceEvent],
            node_id: int,
            frame: FrameType,
            **kwargs,
        ):
            non_str_event = pyc.TraceEvent(event)
            if frame.f_code.co_filename == _FILENAME and non_str_event in events:
                _RECORDED_EVENTS.append(non_str_event)
            return original_emit_event(self, event, node_id, frame, **kwargs)

        try:
            pyc.BaseTracer.__init__ = patched_init
            pyc.BaseTracer._emit_event = _patched_emit_event
            _RECORDED_EVENTS.clear()
            testfunc(events)
        finally:
            pyc.BaseTracer.__init__ = orig_init
            pyc.BaseTracer._emit_event = original_emit_event

    return wrapped_testfunc


def filter_events_to_subset(
    events: List[pyc.TraceEvent], subset: Set[pyc.TraceEvent]
) -> List[pyc.TraceEvent]:
    return [evt for evt in events if evt in subset]


def throw_and_print_diff_if_recorded_not_equal_to(actual: List[pyc.TraceEvent]) -> None:
    assert _RECORDED_EVENTS == actual, "\n".join(
        _DIFFER.compare(
            [evt.value for evt in _RECORDED_EVENTS], [evt.value for evt in actual]
        )
    )
    _RECORDED_EVENTS.clear()


@st.composite
def subsets(draw, elements):
    return {e for e in elements if draw(st.booleans())}


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_recorded_events_simple(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec('logging.debug("foo")', filename=_FILENAME)
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_attribute_load,
                pyc.after_attribute_load,
                pyc.before_call,
                pyc.before_argument,
                pyc.after_string,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_expr_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(pyc.TraceEvent))
@patch_events_and_emitter
def test_recorded_events_two_stmts(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        x = [1, 2, 3]
        logging.debug(x)
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.before_list_literal,
                *([pyc.after_int, pyc.list_elt] * 3),
                pyc.after_list_literal,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.before_stmt,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_attribute_load,
                pyc.after_attribute_load,
                pyc.before_call,
                pyc.before_argument,
                pyc.load_name,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_expr_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_nested_chains_no_call(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        'logging.debug("foo is %s", logging.debug("foo"))', filename=_FILENAME
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_attribute_load,
                pyc.after_attribute_load,
                pyc.before_call,
                pyc.before_argument,
                pyc.after_string,
                pyc.after_argument,
                # next events correspond to `logging.info("foo")`
                pyc.before_argument,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_attribute_load,
                pyc.after_attribute_load,
                pyc.before_call,
                pyc.before_argument,
                pyc.after_string,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_expr_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(pyc.TraceEvent))
@patch_events_and_emitter
def test_list_nested_in_dict(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec("x = {1: [2, 3, 4]}", filename=_FILENAME)
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.before_dict_literal,
                pyc.after_int,
                pyc.dict_key,
                pyc.before_list_literal,
                *([pyc.after_int, pyc.list_elt] * 3),
                pyc.after_list_literal,
                pyc.dict_value,
                pyc.after_dict_literal,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(pyc.TraceEvent))
@patch_events_and_emitter
def test_function_call(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        def foo(x):
            return [x]
        foo([42])
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.before_stmt,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.before_argument,
                pyc.before_list_literal,
                pyc.after_int,
                pyc.list_elt,
                pyc.after_list_literal,
                pyc.after_argument,
                pyc.call,
                pyc.before_function_body,
                pyc.before_stmt,
                pyc.before_return,
                pyc.before_list_literal,
                pyc.load_name,
                pyc.list_elt,
                pyc.after_list_literal,
                pyc.after_return,
                pyc.after_function_execution,
                pyc.return_,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_expr_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_lambda_in_tuple(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec("x: foo = (lambda: 42,)", filename=_FILENAME)
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.before_tuple_literal,
                pyc.before_lambda,
                pyc.after_lambda,
                pyc.tuple_elt,
                pyc.after_tuple_literal,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_decorator(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        def bar(name):
            def decorator(func):
                func.__name__ = name
                return func
            return decorator
        name = "bar"
        @bar(name)
        def foo():
            return 42
        assert foo() == 42
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.after_string,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.before_stmt,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.before_argument,
                pyc.load_name,
                pyc.after_argument,
                pyc.before_function_body,
                pyc.before_stmt,
                pyc.after_stmt,
                pyc.before_stmt,
                pyc.before_return,
                pyc.load_name,
                pyc.after_return,
                pyc.after_function_execution,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.decorator,
                pyc.before_function_body,
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.load_name,
                pyc.after_assign_rhs,
                pyc.load_name,
                pyc.before_attribute_store,
                pyc.after_stmt,
                pyc.before_stmt,
                pyc.before_return,
                pyc.load_name,
                pyc.after_return,
                pyc.after_function_execution,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.before_stmt,
                pyc.before_compare,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.before_function_body,
                pyc.before_stmt,
                pyc.before_return,
                pyc.after_int,
                pyc.after_return,
                pyc.after_function_execution,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.left_compare_arg,
                pyc.after_int,
                pyc.compare_arg,
                pyc.after_compare,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_fundef_defaults(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        "def foo(bar=baz, *, bat=bam): pass",
        global_env={"baz": 42, "bam": 43},
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.load_name,
                pyc.load_name,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_classdef(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        class Foo:
            pass
        class Bar(Foo):
            pass
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_stmt,
                pyc.after_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.before_stmt,
                pyc.load_name,
                pyc.before_stmt,
                pyc.after_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_fancy_slices(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        class Foo:
            def __init__(self, x):
                self.x = x
            def __getitem__(self, x):
                return self
        foo = Foo(1)
        logging.debug(foo[foo.x:foo.x+1,...])
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                # class Foo: ...
                pyc.init_module,
                pyc.before_stmt,
                pyc.call,
                pyc.before_stmt,
                pyc.after_stmt,
                pyc.before_stmt,
                pyc.after_stmt,
                pyc.return_,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                # foo = Foo(1)
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.before_argument,
                pyc.after_int,
                pyc.after_argument,
                pyc.call,
                pyc.before_function_body,
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.load_name,
                pyc.after_assign_rhs,
                pyc.load_name,
                pyc.before_attribute_store,
                pyc.after_stmt,
                pyc.after_function_execution,
                pyc.return_,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                # logging.info(foo[foo.x:foo.x+1,...])
                pyc.before_stmt,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_attribute_load,
                pyc.after_attribute_load,
                pyc.before_call,
                pyc.before_argument,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_subscript_slice,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_attribute_load,
                pyc.after_attribute_load,
                pyc.after_load_complex_symbol,
                pyc.before_binop,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_attribute_load,
                pyc.after_attribute_load,
                pyc.after_load_complex_symbol,
                pyc.left_binop_arg,
                pyc.after_int,
                pyc.right_binop_arg,
                pyc.after_binop,
                pyc.ellipses,
                pyc.after_subscript_slice,
                pyc.before_subscript_load,
                TraceEvent._load_saved_slice,
                pyc.call,
                pyc.before_function_body,
                pyc.before_stmt,
                pyc.before_return,
                pyc.load_name,
                pyc.after_return,
                pyc.after_function_execution,
                pyc.return_,
                pyc.after_subscript_load,
                pyc.after_load_complex_symbol,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_expr_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_for_loop(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        for i in range(10):
            pass
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_for_iter,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.before_argument,
                pyc.after_int,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_for_iter,
            ]
            + [
                pyc.before_for_loop_body,
                pyc.before_stmt,
                pyc.after_stmt,
                pyc.after_for_loop_iter,
            ]
            * 10
            + [
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_while_loop(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        i = 0
        while i < 10:
            i += 1
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.after_int,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.before_stmt,
            ]
            + [
                pyc.before_compare,
                pyc.load_name,
                pyc.left_compare_arg,
                pyc.after_int,
                pyc.compare_arg,
                pyc.after_compare,
                pyc.after_while_test,
                pyc.before_while_loop_body,
                pyc.before_stmt,
                pyc.before_augassign_rhs,
                pyc.after_int,
                pyc.after_augassign_rhs,
                pyc.after_stmt,
                pyc.after_while_loop_iter,
            ]
            * 10
            + [
                pyc.before_compare,
                pyc.load_name,
                pyc.left_compare_arg,
                pyc.after_int,
                pyc.compare_arg,
                pyc.after_compare,
                pyc.after_while_test,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_loop_with_continue(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        for i in range(10):
            continue
            print("hi")
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_for_iter,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.before_argument,
                pyc.after_int,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_for_iter,
            ]
            + [
                pyc.before_for_loop_body,
                pyc.before_stmt,
                pyc.after_for_loop_iter,
            ]
            * 10
            + [
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_for_loop_nested_in_while_loop(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        i = 0
        while i < 10:
            for j in range(2):
                i += 1
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.after_int,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.before_stmt,
            ]
            + [
                pyc.before_compare,
                pyc.load_name,
                pyc.left_compare_arg,
                pyc.after_int,
                pyc.compare_arg,
                pyc.after_compare,
                pyc.after_while_test,
                pyc.before_while_loop_body,
                pyc.before_stmt,
                pyc.before_for_iter,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.before_argument,
                pyc.after_int,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_for_iter,
                *(
                    [
                        pyc.before_for_loop_body,
                        pyc.before_stmt,
                        pyc.before_augassign_rhs,
                        pyc.after_int,
                        pyc.after_augassign_rhs,
                        pyc.after_stmt,
                        pyc.after_for_loop_iter,
                    ]
                    * 2
                ),
                pyc.after_stmt,
                pyc.after_while_loop_iter,
            ]
            * 5
            + [
                pyc.before_compare,
                pyc.load_name,
                pyc.left_compare_arg,
                pyc.after_int,
                pyc.compare_arg,
                pyc.after_compare,
                pyc.after_while_test,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_lambda_wrapping_call(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        z = 42
        def f():
            return z
        lam = lambda: f()
        x = lam()
        """,
        filename=_FILENAME,
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                # z = 42
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.after_int,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                # def f(): ...
                pyc.before_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                # lam = lambda: f()
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.before_lambda,
                pyc.after_lambda,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                # x = lam()
                pyc.before_stmt,
                pyc.before_assign_rhs,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.call,
                pyc.before_lambda_body,
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.call,
                pyc.before_function_body,
                pyc.before_stmt,
                pyc.before_return,
                pyc.load_name,
                pyc.after_return,
                pyc.after_function_execution,
                pyc.return_,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_lambda_body,
                pyc.return_,
                pyc.after_call,
                pyc.after_load_complex_symbol,
                pyc.after_assign_rhs,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_comprehension(events):
    assert _RECORDED_EVENTS == []
    pyc.BaseTracer().exec(
        """
        [i for i in range(10) if i%2==0]
        """,
        filename=_FILENAME,
    )
    if_evts = [
        pyc.before_compare,
        pyc.before_binop,
        pyc.load_name,
        pyc.left_binop_arg,
        pyc.after_int,
        pyc.right_binop_arg,
        pyc.after_binop,
        pyc.left_compare_arg,
        pyc.after_int,
        pyc.compare_arg,
        pyc.after_compare,
        pyc.after_comprehension_if,
    ]
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset(
            [
                pyc.init_module,
                pyc.before_stmt,
                # range
                pyc.before_load_complex_symbol,
                pyc.load_name,
                pyc.before_call,
                pyc.before_argument,
                pyc.after_int,
                pyc.after_argument,
                pyc.after_call,
                pyc.after_load_complex_symbol,
            ]
            + [
                *if_evts,
                pyc.load_name,
                pyc.after_comprehension_elt,
                *if_evts,
            ]
            * 5
            + [
                pyc.after_expr_stmt,
                pyc.after_stmt,
                TraceEvent._load_saved_expr_stmt_ret,
                pyc.after_module_stmt,
                pyc.exit_module,
            ],
            events,
        )
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_and_emitter
def test_exception_handler_types(events):
    for exc_type in "", " AssertionError":
        assert _RECORDED_EVENTS == []
        pyc.BaseTracer().exec(
            f"""
            try:
                assert False
            except{exc_type}:
                ...
            """,
            filename=_FILENAME,
        )
        throw_and_print_diff_if_recorded_not_equal_to(
            filter_events_to_subset(
                [
                    pyc.init_module,
                    pyc.before_stmt,  # try...
                    pyc.before_stmt,  # assert False
                    pyc.after_bool,
                    *([pyc.load_name] if exc_type else []),
                    pyc.exception_handler_type,
                    pyc.before_stmt,
                    pyc.ellipses,
                    pyc.after_expr_stmt,
                    pyc.after_stmt,
                    pyc.after_stmt,
                    TraceEvent._load_saved_expr_stmt_ret,
                    pyc.after_module_stmt,
                    pyc.exit_module,
                ],
                events,
            )
        )
