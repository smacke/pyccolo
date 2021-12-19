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
    pyc.BaseTracerStateMachine.clear_instance()
    pyc.BaseTracerStateMachine._emit_event = _patched_emit_event
    yield
    pyc.BaseTracerStateMachine._emit_event = original_emit_event


_DIFFER = difflib.Differ()


def patch_events_with_registered_handlers_to_subset(testfunc):

    @functools.wraps(testfunc)
    @settings(max_examples=20, deadline=None)
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

        orig_handlers = pyc.BaseTracerStateMachine.instance().events_with_registered_handlers
        try:
            pyc.BaseTracerStateMachine.instance().events_with_registered_handlers = frozenset(events)
            _RECORDED_EVENTS.clear()
            testfunc(events)
        finally:
            pyc.BaseTracerStateMachine.instance().events_with_registered_handlers = orig_handlers

    return wrapped_testfunc


def filter_events_to_subset(events: List[pyc.TraceEvent], subset: Set[pyc.TraceEvent]) -> List[pyc.TraceEvent]:
    return [evt for evt in events if evt in subset]


def throw_and_print_diff_if_recorded_not_equal_to(actual: List[pyc.TraceEvent]) -> None:
    assert _RECORDED_EVENTS == actual, '\n'.join(
        _DIFFER.compare([evt.value for evt in _RECORDED_EVENTS], [evt.value for evt in actual])
    )
    _RECORDED_EVENTS.clear()


def run_cell(cell, **kwargs):
    pyc.BaseTracerStateMachine.instance().exec_sandboxed(cell)


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


# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_nested_chains_no_call(events):
#     assert _RECORDED_EVENTS == []
#     run_cell('logging.info("foo is %s", logging.info("foo"))')
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_attribute_load,
#             TraceEvent.before_call,
#             TraceEvent.argument,
#
#             # next events correspond to `logging.info("foo")`
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_attribute_load,
#             TraceEvent.before_call,
#             TraceEvent.argument,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.argument,
#
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_list_nested_in_dict(events):
#     assert _RECORDED_EVENTS == []
#     run_cell('x = {1: [2, 3, 4]}')
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_assign_rhs,
#             TraceEvent.before_dict_literal,
#
#             TraceEvent.dict_key,
#             TraceEvent.before_list_literal,
#             *([TraceEvent.list_elt] * 3),
#             TraceEvent.after_list_literal,
#             TraceEvent.dict_value,
#
#             TraceEvent.after_dict_literal,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_function_call(events):
#     assert _RECORDED_EVENTS == []
#     run_cell(
#         """
#         def foo(x):
#             return [x]
#         """
#     )
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#     run_cell('foo([42])')
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_call,
#             TraceEvent.before_list_literal,
#             TraceEvent.list_elt,
#             TraceEvent.after_list_literal,
#             TraceEvent.argument,
#             TraceEvent.call,
#             TraceEvent.before_function_body,
#             TraceEvent.before_stmt,
#             TraceEvent.before_return,
#             TraceEvent.before_list_literal,
#             TraceEvent.load_name,
#             TraceEvent.list_elt,
#             TraceEvent.after_list_literal,
#             TraceEvent.after_return,
#             TraceEvent.after_function_execution,
#             TraceEvent.return_,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_lambda_in_tuple(events):
#     assert _RECORDED_EVENTS == []
#     run_cell('x = (lambda: 42,)')
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_assign_rhs,
#             TraceEvent.before_tuple_literal,
#             TraceEvent.before_lambda,
#             TraceEvent.after_lambda,
#             TraceEvent.tuple_elt,
#             TraceEvent.after_tuple_literal,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_fancy_slices(events):
#     assert _RECORDED_EVENTS == []
#     run_cell(
#         """
#         import numpy as np
#         class Foo:
#             def __init__(self, x):
#                 self.x = x
#         foo = Foo(1)
#         arr = np.zeros((3, 3, 3))
#         """
#     )
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             # import numpy as np
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#
#             # class Foo: ...
#             TraceEvent.before_stmt,
#             TraceEvent.call,
#             TraceEvent.before_stmt,
#             TraceEvent.after_stmt,
#             TraceEvent.return_,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#
#             # foo = Foo(1)
#             TraceEvent.before_stmt,
#             TraceEvent.before_assign_rhs,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_call,
#             TraceEvent.argument,
#             TraceEvent.call,
#             TraceEvent.before_function_body,
#             TraceEvent.before_stmt,
#             TraceEvent.before_assign_rhs,
#             TraceEvent.load_name,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.load_name,
#             TraceEvent.before_attribute_store,
#             TraceEvent.after_stmt,
#             TraceEvent.after_function_execution,
#             TraceEvent.return_,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#
#             # arr = np.zeros((3, 3, 3))
#             TraceEvent.before_stmt,
#             TraceEvent.before_assign_rhs,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_attribute_load,
#             TraceEvent.before_call,
#             TraceEvent.before_tuple_literal,
#             TraceEvent.tuple_elt,
#             TraceEvent.tuple_elt,
#             TraceEvent.tuple_elt,
#             TraceEvent.after_tuple_literal,
#             TraceEvent.argument,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#     run_cell('logging.info(arr[foo.x:foo.x+1,...])')
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_attribute_load,
#             TraceEvent.before_call,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_attribute_load,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_attribute_load,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.subscript_slice,
#             TraceEvent.before_subscript_load,
#             TraceEvent._load_saved_slice,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.argument,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_for_loop(events):
#     assert _RECORDED_EVENTS == []
#     run_cell(
#         """
#         for i in range(10):
#             pass
#         """
#     )
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_call,
#             TraceEvent.argument,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#         ] + [
#             TraceEvent.before_for_loop_body,
#             TraceEvent.before_stmt,
#             TraceEvent.after_stmt,
#             TraceEvent.after_for_loop_iter,
#         # ] * 10 + [
#         ] * 1 + [
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_while_loop(events):
#     assert _RECORDED_EVENTS == []
#     run_cell(
#         """
#         i = 0
#         while i < 10:
#             i += 1
#         """
#     )
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_assign_rhs,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#             TraceEvent.before_stmt,
#         ] + [
#             TraceEvent.load_name,
#             TraceEvent.before_while_loop_body,
#             TraceEvent.before_stmt,
#             TraceEvent.after_stmt,
#             TraceEvent.after_while_loop_iter,
#         # ] * 10 + [
#         ] * 1 + [
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_loop_with_continue(events):
#     assert _RECORDED_EVENTS == []
#     run_cell(
#         """
#         for i in range(10):
#             continue
#             print("hi")
#         """
#     )
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_call,
#             TraceEvent.argument,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.before_for_loop_body,
#             TraceEvent.before_stmt,
#             TraceEvent.after_for_loop_iter,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_for_loop_nested_in_while_loop(events):
#     assert _RECORDED_EVENTS == []
#     run_cell(
#         """
#         i = 0
#         while i < 10:
#             for j in range(2):
#                 i += 1
#         """
#     )
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#             TraceEvent.before_stmt,
#             TraceEvent.before_assign_rhs,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#             TraceEvent.before_stmt,
#         ] + [
#             TraceEvent.load_name,
#             TraceEvent.before_while_loop_body,
#             TraceEvent.before_stmt,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_call,
#             TraceEvent.argument,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#
#             TraceEvent.before_for_loop_body,
#             TraceEvent.before_stmt,
#             TraceEvent.after_stmt,
#             TraceEvent.after_for_loop_iter,
#             # TraceEvent.before_stmt,
#             # TraceEvent.after_stmt,
#             # TraceEvent.after_loop_iter,
#
#             TraceEvent.after_stmt,
#             TraceEvent.after_while_loop_iter,
#         # ] * 5 + [
#         ] * 1 + [
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
#
#
# @given(events=subsets(_ALL_EVENTS_WITH_HANDLERS))
# @patch_events_with_registered_handlers_to_subset
# def test_lambda_wrapping_call(events):
#     assert _RECORDED_EVENTS == []
#     run_cell(
#         """
#         z = 42
#         def f():
#             return z
#         lam = lambda: f()
#         x = lam()
#         """
#     )
#     throw_and_print_diff_if_recorded_not_equal_to(
#         filter_events_to_subset([
#             TraceEvent.init_module,
#
#             # z = 42
#             TraceEvent.before_stmt,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#
#             # def f(): ...
#             TraceEvent.before_stmt,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#
#             # lam = lambda: f()
#             TraceEvent.before_stmt,
#             TraceEvent.before_lambda,
#             TraceEvent.after_lambda,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#
#             # x = lam()
#             TraceEvent.before_stmt,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_call,
#             TraceEvent.call,
#             TraceEvent.before_lambda_body,
#             TraceEvent.before_load_complex_symbol,
#             TraceEvent.load_name,
#             TraceEvent.before_call,
#             TraceEvent.call,
#             TraceEvent.before_function_body,
#             TraceEvent.before_stmt,
#             TraceEvent.before_return,
#             TraceEvent.load_name,
#             TraceEvent.after_return,
#             TraceEvent.after_function_execution,
#             TraceEvent.return_,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.return_,
#             TraceEvent.after_call,
#             TraceEvent.after_load_complex_symbol,
#             TraceEvent.after_assign_rhs,
#             TraceEvent.after_stmt,
#             TraceEvent.after_module_stmt,
#         ], events)
#     )
