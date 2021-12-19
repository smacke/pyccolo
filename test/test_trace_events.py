# -*- coding: future_annotations -*-
import difflib
import functools
import logging
import sys
from typing import TYPE_CHECKING

import hypothesis.strategies as st
import pytest
from hypothesis import example, given, settings

import pyccolo as pyc

if TYPE_CHECKING:
    from typing import List, Set, Union
    from types import FrameType

logging.basicConfig(level=logging.INFO)


_RECORDED_EVENTS = []


def subsets(draw, elements):
    return {e for e in elements if draw(st.booleans())}


@pytest.fixture(autouse=True)
def patched_emit_event_fixture():
    _RECORDED_EVENTS.clear()
    original_emit_event = pyc.BaseTracerStateMachine._emit_event

    def _patched_emit_event(self, evt: Union[pyc.TraceEvent, str], *args, **kwargs):
        event = pyc.TraceEvent(evt) if isinstance(evt, str) else evt
        frame: FrameType = kwargs.get('_frame', sys._getframe().f_back)
        kwargs['_frame'] = frame
        if frame.f_code.co_filename == '<sandbox>':
            _RECORDED_EVENTS.append(event)
        return original_emit_event(self, evt, *args, **kwargs)
    pyc.clear_instance()
    pyc.BaseTracerStateMachine._emit_event = _patched_emit_event
    yield
    pyc.BaseTracerStateMachine._emit_event = original_emit_event


_DIFFER = difflib.Differ()


def patch_events_with_registered_handlers_to_subset(testfunc):

    @functools.wraps(testfunc)
    @settings(max_examples=25, deadline=None)
    @example(events=set(pyc.TraceEvent))
    def wrapped_testfunc(events):
        if events & {pyc.before_subscript_load, pyc.before_subscript_store, pyc.before_subscript_del}:
            events.add(pyc._load_saved_slice)
        if pyc._load_saved_slice in events:
            events.add(pyc.before_subscript_load)
            events.add(pyc.before_subscript_store)
            events.add(pyc.before_subscript_del)
        # get rid of all the builtin tracing events since we'll have all if we have one
        events -= {
            pyc.line, pyc.call, pyc.c_call, pyc.return_, pyc.c_return, pyc.exception, pyc.c_exception, pyc.opcode
        }

        orig_handlers = pyc.tracer().events_with_registered_handlers
        try:
            pyc.tracer().events_with_registered_handlers = frozenset(events)
            _RECORDED_EVENTS.clear()
            testfunc(events)
        finally:
            pyc.tracer().events_with_registered_handlers = orig_handlers

    return wrapped_testfunc


def filter_events_to_subset(events: List[pyc.TraceEvent], subset: Set[pyc.TraceEvent]) -> List[pyc.TraceEvent]:
    return [evt for evt in events if evt in subset]


def throw_and_print_diff_if_recorded_not_equal_to(actual: List[pyc.TraceEvent]) -> None:
    assert _RECORDED_EVENTS == actual, '\n'.join(
        _DIFFER.compare([evt.value for evt in _RECORDED_EVENTS], [evt.value for evt in actual])
    )
    _RECORDED_EVENTS.clear()


def run_cell(cell, **kwargs):
    pyc.tracer().exec_sandboxed(cell)


@st.composite
def subsets(draw, elements):
    return {e for e in elements if draw(st.booleans())}


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_recorded_events_simple(events):
    assert _RECORDED_EVENTS == []
    run_cell('logging.info("foo")')
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_attribute_load,
            pyc.after_attribute_load,
            pyc.before_call,
            pyc.argument,
            pyc.after_call,
            pyc.after_load_complex_symbol,
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(pyc.TraceEvent))
@patch_events_with_registered_handlers_to_subset
def test_recorded_events_two_stmts(events):
    assert _RECORDED_EVENTS == []
    run_cell(
        """
        x = [1, 2, 3]
        logging.info(x)
        """
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.before_assign_rhs,
            pyc.before_list_literal,
            *([pyc.list_elt] * 3),
            pyc.after_list_literal,
            pyc.after_assign_rhs,
            pyc.after_stmt,
            pyc.after_module_stmt,

            pyc.before_stmt,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_attribute_load,
            pyc.after_attribute_load,
            pyc.before_call,
            pyc.load_name,
            pyc.argument,
            pyc.after_call,
            pyc.after_load_complex_symbol,
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_nested_chains_no_call(events):
    assert _RECORDED_EVENTS == []
    run_cell('logging.info("foo is %s", logging.info("foo"))')
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_attribute_load,
            pyc.after_attribute_load,
            pyc.before_call,
            pyc.argument,

            # next events correspond to `logging.info("foo")`
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_attribute_load,
            pyc.after_attribute_load,
            pyc.before_call,
            pyc.argument,
            pyc.after_call,
            pyc.after_load_complex_symbol,
            pyc.argument,

            pyc.after_call,
            pyc.after_load_complex_symbol,
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(pyc.TraceEvent))
@patch_events_with_registered_handlers_to_subset
def test_list_nested_in_dict(events):
    assert _RECORDED_EVENTS == []
    run_cell('x = {1: [2, 3, 4]}')
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.before_assign_rhs,
            pyc.before_dict_literal,

            pyc.dict_key,
            pyc.before_list_literal,
            *([pyc.list_elt] * 3),
            pyc.after_list_literal,
            pyc.dict_value,

            pyc.after_dict_literal,
            pyc.after_assign_rhs,
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(pyc.TraceEvent))
@patch_events_with_registered_handlers_to_subset
def test_function_call(events):
    assert _RECORDED_EVENTS == []
    run_cell(
        """
        def foo(x):
            return [x]
        foo([42])
        """
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.after_stmt,
            pyc.after_module_stmt,

            pyc.before_stmt,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_call,
            pyc.before_list_literal,
            pyc.list_elt,
            pyc.after_list_literal,
            pyc.argument,
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
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_lambda_in_tuple(events):
    assert _RECORDED_EVENTS == []
    run_cell('x = (lambda: 42,)')
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
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
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_fancy_slices(events):
    assert _RECORDED_EVENTS == []
    run_cell(
        """
        import numpy as np
        class Foo:
            def __init__(self, x):
                self.x = x
        foo = Foo(1)
        arr = np.zeros((3, 3, 3))
        logging.info(arr[foo.x:foo.x+1,...])
        """
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            # import numpy as np
            pyc.init_module,
            pyc.before_stmt,
            pyc.after_stmt,
            pyc.after_module_stmt,

            # class Foo: ...
            pyc.before_stmt,
            pyc.call,
            pyc.before_stmt,
            pyc.after_stmt,
            pyc.return_,
            pyc.after_stmt,
            pyc.after_module_stmt,

            # foo = Foo(1)
            pyc.before_stmt,
            pyc.before_assign_rhs,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_call,
            pyc.argument,
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
            pyc.after_module_stmt,

            # arr = np.zeros((3, 3, 3))
            pyc.before_stmt,
            pyc.before_assign_rhs,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_attribute_load,
            pyc.after_attribute_load,
            pyc.before_call,
            pyc.before_tuple_literal,
            pyc.tuple_elt,
            pyc.tuple_elt,
            pyc.tuple_elt,
            pyc.after_tuple_literal,
            pyc.argument,
            pyc.after_call,
            pyc.after_load_complex_symbol,
            pyc.after_assign_rhs,
            pyc.after_stmt,
            pyc.after_module_stmt,

            pyc.before_stmt,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_attribute_load,
            pyc.after_attribute_load,
            pyc.before_call,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_attribute_load,
            pyc.after_attribute_load,
            pyc.after_load_complex_symbol,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_attribute_load,
            pyc.after_attribute_load,
            pyc.after_load_complex_symbol,
            pyc.left_binop_arg,
            pyc.right_binop_arg,
            pyc.add,
            pyc.subscript_slice,
            pyc.before_subscript_load,
            pyc._load_saved_slice,
            pyc.after_subscript_load,
            pyc.after_load_complex_symbol,
            pyc.argument,
            pyc.after_call,
            pyc.after_load_complex_symbol,
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_for_loop(events):
    assert _RECORDED_EVENTS == []
    run_cell(
        """
        for i in range(10):
            pass
        """
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_call,
            pyc.argument,
            pyc.after_call,
            pyc.after_load_complex_symbol,
        ] + [
            pyc.before_for_loop_body,
            pyc.before_stmt,
            pyc.after_stmt,
            pyc.after_for_loop_iter,
        ] * 10 + [
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_while_loop(events):
    assert _RECORDED_EVENTS == []
    run_cell(
        """
        i = 0
        while i < 10:
            i += 1
        """
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.before_assign_rhs,
            pyc.after_assign_rhs,
            pyc.after_stmt,
            pyc.after_module_stmt,
            pyc.before_stmt,
        ] + [
            pyc.load_name,
            pyc.before_while_loop_body,
            pyc.before_stmt,
            pyc.after_stmt,
            pyc.after_while_loop_iter,
        ] * 10 + [
            pyc.load_name,
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_loop_with_continue(events):
    assert _RECORDED_EVENTS == []
    run_cell(
        """
        for i in range(10):
            continue
            print("hi")
        """
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_call,
            pyc.argument,
            pyc.after_call,
            pyc.after_load_complex_symbol,
        ] + [
            pyc.before_for_loop_body,
            pyc.before_stmt,
            pyc.after_for_loop_iter,
        ] * 10 + [
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_for_loop_nested_in_while_loop(events):
    assert _RECORDED_EVENTS == []
    run_cell(
        """
        i = 0
        while i < 10:
            for j in range(2):
                i += 1
        """
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,
            pyc.before_stmt,
            pyc.before_assign_rhs,
            pyc.after_assign_rhs,
            pyc.after_stmt,
            pyc.after_module_stmt,
            pyc.before_stmt,
        ] + [
            pyc.load_name,
            pyc.before_while_loop_body,
            pyc.before_stmt,
            pyc.before_load_complex_symbol,
            pyc.load_name,
            pyc.before_call,
            pyc.argument,
            pyc.after_call,
            pyc.after_load_complex_symbol,

            *([
                pyc.before_for_loop_body,
                pyc.before_stmt,
                pyc.after_stmt,
                pyc.after_for_loop_iter,
            ] * 2),

            pyc.after_stmt,
            pyc.after_while_loop_iter,
        ] * 5 + [
            pyc.load_name,
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )


@given(events=subsets(set(pyc.TraceEvent)))
@patch_events_with_registered_handlers_to_subset
def test_lambda_wrapping_call(events):
    assert _RECORDED_EVENTS == []
    run_cell(
        """
        z = 42
        def f():
            return z
        lam = lambda: f()
        x = lam()
        """
    )
    throw_and_print_diff_if_recorded_not_equal_to(
        filter_events_to_subset([
            pyc.init_module,

            # z = 42
            pyc.before_stmt,
            pyc.before_assign_rhs,
            pyc.after_assign_rhs,
            pyc.after_stmt,
            pyc.after_module_stmt,

            # def f(): ...
            pyc.before_stmt,
            pyc.after_stmt,
            pyc.after_module_stmt,

            # lam = lambda: f()
            pyc.before_stmt,
            pyc.before_assign_rhs,
            pyc.before_lambda,
            pyc.after_lambda,
            pyc.after_assign_rhs,
            pyc.after_stmt,
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
            pyc.return_,
            pyc.after_call,
            pyc.after_load_complex_symbol,
            pyc.after_assign_rhs,
            pyc.after_stmt,
            pyc.after_module_stmt,
        ], events)
    )
