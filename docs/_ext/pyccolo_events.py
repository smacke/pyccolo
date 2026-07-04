# -*- coding: utf-8 -*-
"""Sphinx extension that renders Pyccolo's event taxonomy straight from the code.

The ``.. pyccolo-events::`` directive introspects
:class:`pyccolo.trace_events.TraceEvent` (plus the ``BEFORE_EXPR_EVENTS``,
``SYS_TRACE_EVENTS`` and ``AST_TO_EVENT_MAPPING`` tables) and emits one table per
category. Per-event prose lives in the curated ``EVENTS`` mapping below; the flag
columns (thunk / sys / ast) are derived from the source sets so they can never
drift from the library.

A build-time completeness check asserts that every public ``TraceEvent`` member
is documented here and slotted into exactly one category. Adding an event to
``pyccolo/trace_events.py`` without describing it here therefore fails the docs
build, keeping this reference in lockstep with the code.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList

from pyccolo.trace_events import (
    AST_TO_EVENT_MAPPING,
    BEFORE_EXPR_EVENTS,
    SYS_TRACE_EVENTS,
    TraceEvent,
)

# name -> (fires on, what a handler's return value does)
EVENTS: Dict[str, Tuple[str, str]] = {
    # Module & import lifecycle
    "before_import": ("an ``import`` statement is about to run", "observe only"),
    "init_module": ("a module body begins executing", "observe only"),
    "exit_module": ("a module body finishes executing", "observe only"),
    "after_import": ("an ``import`` statement has run", "observe only"),
    # Statements
    "before_stmt": (
        "each statement, before it runs",
        "a thunk to run instead, or ``pyc.Pass`` to skip the statement",
    ),
    "after_stmt": ("each statement, after it runs", "observe only"),
    "after_module_stmt": (
        "a top-level module statement finishes",
        "replacement value (the displayed result)",
    ),
    "after_expr_stmt": ("a bare expression statement finishes", "replacement value"),
    "before_assert": ("an ``assert`` statement, before it runs", "observe only"),
    "after_assert": ("an ``assert`` statement, after it runs", "observe only"),
    # Names & literals
    "load_name": ("a bare name is loaded (e.g. ``foo``)", "replacement value"),
    "after_bool": ("a ``True`` / ``False`` literal is evaluated", "replacement value"),
    "after_int": ("an ``int`` literal is evaluated", "replacement value"),
    "after_float": ("a ``float`` literal is evaluated", "replacement value"),
    "after_complex": ("a ``complex`` literal is evaluated", "replacement value"),
    "after_string": ("a ``str`` literal is evaluated", "replacement value"),
    "after_bytes": ("a ``bytes`` literal is evaluated", "replacement value"),
    "after_none": ("a ``None`` literal is evaluated", "replacement value"),
    "ellipsis": ("an ``...`` (Ellipsis) literal is evaluated", "replacement value"),
    "before_fstring": ("an f-string, before it is built", "callable (thunk)"),
    "after_fstring": ("an f-string, after it is built", "replacement value"),
    # Assignment
    "before_assign_rhs": (
        "the right-hand side of an assignment, before it runs",
        "callable (thunk)",
    ),
    "after_assign_rhs": (
        "the right-hand side of an assignment, after it runs",
        "replacement value",
    ),
    "before_augassign_rhs": (
        "the RHS of an augmented assignment (``+=`` etc.), before it runs",
        "callable (thunk)",
    ),
    "after_augassign_rhs": (
        "the RHS of an augmented assignment, after it runs",
        "replacement value",
    ),
    # Attributes & subscripts
    "before_attribute_load": (
        "an attribute load ``obj.attr``, before dereference",
        "the receiver object to dereference",
    ),
    "after_attribute_load": (
        "an attribute load, after dereference",
        "replacement value",
    ),
    "before_attribute_store": ("an attribute store ``obj.attr = ...``", "observe only"),
    "before_attribute_del": ("an attribute delete ``del obj.attr``", "observe only"),
    "before_subscript_load": (
        "a subscript load ``obj[key]``, before dereference",
        "the receiver object to dereference",
    ),
    "after_subscript_load": (
        "a subscript load, after dereference",
        "replacement value",
    ),
    "before_subscript_store": ("a subscript store ``obj[key] = ...``", "observe only"),
    "before_subscript_del": ("a subscript delete ``del obj[key]``", "observe only"),
    "before_subscript_slice": (
        "the slice/key of a subscript, before it is evaluated",
        "callable (thunk)",
    ),
    "after_subscript_slice": (
        "the slice/key of a subscript, after it is evaluated",
        "replacement value",
    ),
    "before_load_complex_symbol": (
        "a compound load (chained attribute/subscript/call), before it runs",
        "callable (thunk)",
    ),
    "after_load_complex_symbol": (
        "a compound load, after it runs",
        "replacement value",
    ),
    # Calls & arguments
    "decorator": ("a decorator is applied", "replacement value (the decorator)"),
    "before_call": ("a call ``f(...)``, before invocation", "the callable to invoke"),
    "after_call": ("a call ``f(...)``, after it returns", "replacement value"),
    "before_argument": ("a call argument, before it is evaluated", "callable (thunk)"),
    "after_argument": ("a call argument, after it is evaluated", "replacement value"),
    "before_return": ("a ``return`` value, before it is evaluated", "callable (thunk)"),
    "after_return": ("a ``return`` value, after it is evaluated", "replacement value"),
    # Operators
    "before_binop": ("a binary op ``x + y``, before it runs", "callable (thunk)"),
    "after_binop": ("a binary op, after it runs", "replacement value (result)"),
    "before_left_binop_arg": (
        "the left operand of a binop, before it runs",
        "callable (thunk)",
    ),
    "after_left_binop_arg": (
        "the left operand of a binop, after it runs",
        "replacement value",
    ),
    "before_right_binop_arg": (
        "the right operand of a binop, before it runs",
        "callable (thunk)",
    ),
    "after_right_binop_arg": (
        "the right operand of a binop, after it runs",
        "replacement value",
    ),
    "before_unaryop": (
        "a unary op ``-x`` / ``not x``, before it runs",
        "callable (thunk)",
    ),
    "after_unaryop": ("a unary op, after it runs", "replacement value (result)"),
    "before_unaryop_arg": (
        "the operand of a unary op, before it runs",
        "callable (thunk)",
    ),
    "after_unaryop_arg": (
        "the operand of a unary op, after it runs",
        "replacement value",
    ),
    "before_boolop": (
        "a boolean op ``x and y`` / ``x or y``, before it runs",
        "callable (thunk)",
    ),
    "after_boolop": ("a boolean op, after it runs", "replacement value (result)"),
    "before_boolop_arg": (
        "an operand of a boolean op, before it runs",
        "callable (thunk)",
    ),
    "after_boolop_arg": (
        "an operand of a boolean op, after it runs",
        "replacement value",
    ),
    "before_compare": ("a comparison ``x < y``, before it runs", "callable (thunk)"),
    "after_compare": ("a comparison, after it runs", "replacement value (result)"),
    "left_compare_arg": ("the left operand of a comparison", "replacement value"),
    "compare_arg": ("a right-hand operand of a comparison", "replacement value"),
    # Collection literals
    "before_list_literal": (
        "a list literal ``[...]``, before it is built",
        "callable (thunk)",
    ),
    "after_list_literal": ("a list literal, after it is built", "replacement value"),
    "before_tuple_literal": (
        "a tuple literal ``(...)``, before it is built",
        "callable (thunk)",
    ),
    "after_tuple_literal": ("a tuple literal, after it is built", "replacement value"),
    "before_set_literal": (
        "a set literal ``{...}``, before it is built",
        "callable (thunk)",
    ),
    "after_set_literal": ("a set literal, after it is built", "replacement value"),
    "before_dict_literal": (
        "a dict literal ``{k: v}``, before it is built",
        "callable (thunk)",
    ),
    "after_dict_literal": ("a dict literal, after it is built", "replacement value"),
    "list_elt": ("an element of a list literal", "replacement value"),
    "tuple_elt": ("an element of a tuple literal", "replacement value"),
    "set_elt": ("an element of a set literal", "replacement value"),
    "dict_key": ("a key of a dict literal", "replacement value"),
    "dict_value": ("a value of a dict literal", "replacement value"),
    # Comprehensions
    "after_comprehension_if": (
        "a comprehension ``if`` predicate is evaluated",
        "replacement value",
    ),
    "after_comprehension_elt": (
        "a comprehension element is evaluated",
        "replacement value",
    ),
    "after_dict_comprehension_key": (
        "a dict-comprehension key is evaluated",
        "replacement value",
    ),
    "after_dict_comprehension_value": (
        "a dict-comprehension value is evaluated",
        "replacement value",
    ),
    # Control flow
    "after_if_test": (
        "the test of an ``if`` is evaluated",
        "replacement value (test result)",
    ),
    "after_while_test": (
        "the test of a ``while`` is evaluated",
        "replacement value (test result)",
    ),
    "before_for_loop_body": (
        "a ``for`` loop body, before each iteration",
        "observe only",
    ),
    "after_for_loop_iter": (
        "a ``for`` loop body, after each iteration",
        "observe only (guarded)",
    ),
    "before_while_loop_body": (
        "a ``while`` loop body, before each iteration",
        "observe only",
    ),
    "after_while_loop_iter": (
        "a ``while`` loop body, after each iteration",
        "observe only (guarded)",
    ),
    "before_for_iter": (
        "the iterable of a ``for`` loop, before iteration",
        "callable (thunk)",
    ),
    "after_for_iter": (
        "the iterable of a ``for`` loop, after iteration",
        "replacement value",
    ),
    # Functions & lambdas
    "before_function_body": ("a function body, before it executes", "observe only"),
    "after_function_execution": (
        "a function body, after it executes",
        "observe only (guarded)",
    ),
    "before_lambda": (
        "a ``lambda`` expression, before it is created",
        "callable (thunk)",
    ),
    "after_lambda": (
        "a ``lambda`` expression, after it is created",
        "replacement value",
    ),
    "before_lambda_body": ("a ``lambda`` body, before it executes", "observe only"),
    "after_lambda_body": ("a ``lambda`` body, after it executes", "replacement value"),
    # Exceptions
    "exception_handler_type": (
        "the type in an ``except T:`` clause is evaluated",
        "replacement value (the caught type)",
    ),
    # sys.settrace events
    "line": ("``sys.settrace`` fires for a new source line", "observe only"),
    "call": ("a stack frame is pushed (``sys.settrace``)", "observe only"),
    "return_": ("a stack frame is popped (``sys.settrace``)", "observe only"),
    "exception": ("an exception is raised (``sys.settrace``)", "observe only"),
    "opcode": ("a bytecode opcode executes (opt-in, ``sys.settrace``)", "observe only"),
    "c_call": (
        "a C function is called (``sys.settrace``)",
        "observe only (rarely used)",
    ),
    "c_return": (
        "a C function returns (``sys.settrace``)",
        "observe only (rarely used)",
    ),
    "c_exception": (
        "a C function raises (``sys.settrace``)",
        "observe only (rarely used)",
    ),
}

# Ordered (title, [event names]); every public event must live in exactly one.
CATEGORIES: List[Tuple[str, List[str]]] = [
    (
        "Module & import lifecycle",
        [
            "before_import",
            "init_module",
            "exit_module",
            "after_import",
        ],
    ),
    (
        "Statements",
        [
            "before_stmt",
            "after_stmt",
            "after_module_stmt",
            "after_expr_stmt",
            "before_assert",
            "after_assert",
        ],
    ),
    (
        "Names & literals",
        [
            "load_name",
            "after_bool",
            "after_int",
            "after_float",
            "after_complex",
            "after_string",
            "after_bytes",
            "after_none",
            "ellipsis",
            "before_fstring",
            "after_fstring",
        ],
    ),
    (
        "Assignment",
        [
            "before_assign_rhs",
            "after_assign_rhs",
            "before_augassign_rhs",
            "after_augassign_rhs",
        ],
    ),
    (
        "Attributes & subscripts",
        [
            "before_attribute_load",
            "after_attribute_load",
            "before_attribute_store",
            "before_attribute_del",
            "before_subscript_load",
            "after_subscript_load",
            "before_subscript_store",
            "before_subscript_del",
            "before_subscript_slice",
            "after_subscript_slice",
            "before_load_complex_symbol",
            "after_load_complex_symbol",
        ],
    ),
    (
        "Calls & arguments",
        [
            "decorator",
            "before_call",
            "after_call",
            "before_argument",
            "after_argument",
            "before_return",
            "after_return",
        ],
    ),
    (
        "Operators",
        [
            "before_binop",
            "after_binop",
            "before_left_binop_arg",
            "after_left_binop_arg",
            "before_right_binop_arg",
            "after_right_binop_arg",
            "before_unaryop",
            "after_unaryop",
            "before_unaryop_arg",
            "after_unaryop_arg",
            "before_boolop",
            "after_boolop",
            "before_boolop_arg",
            "after_boolop_arg",
            "before_compare",
            "after_compare",
            "left_compare_arg",
            "compare_arg",
        ],
    ),
    (
        "Collection literals",
        [
            "before_list_literal",
            "after_list_literal",
            "before_tuple_literal",
            "after_tuple_literal",
            "before_set_literal",
            "after_set_literal",
            "before_dict_literal",
            "after_dict_literal",
            "list_elt",
            "tuple_elt",
            "set_elt",
            "dict_key",
            "dict_value",
        ],
    ),
    (
        "Comprehensions",
        [
            "after_comprehension_if",
            "after_comprehension_elt",
            "after_dict_comprehension_key",
            "after_dict_comprehension_value",
        ],
    ),
    (
        "Control flow",
        [
            "after_if_test",
            "after_while_test",
            "before_for_loop_body",
            "after_for_loop_iter",
            "before_while_loop_body",
            "after_while_loop_iter",
            "before_for_iter",
            "after_for_iter",
        ],
    ),
    (
        "Functions & lambdas",
        [
            "before_function_body",
            "after_function_execution",
            "before_lambda",
            "after_lambda",
            "before_lambda_body",
            "after_lambda_body",
        ],
    ),
    ("Exceptions", ["exception_handler_type"]),
    (
        "sys.settrace events",
        [
            "line",
            "call",
            "return_",
            "exception",
            "opcode",
            "c_call",
            "c_return",
            "c_exception",
        ],
    ),
]


def _public_events() -> List[str]:
    return [e.name for e in TraceEvent if not e.name.startswith("_")]


def _ast_hooks() -> Dict[str, str]:
    hooks: Dict[str, str] = {}
    for node_type, evt in AST_TO_EVENT_MAPPING.items():
        # keep the first / most specific mapping if several nodes share an event
        hooks.setdefault(evt.name, node_type.__name__)
    return hooks


def _check_coverage() -> None:
    """Fail the build if the code and this reference have drifted apart."""
    public = set(_public_events())
    documented = set(EVENTS)
    categorized: List[str] = [name for _, names in CATEGORIES for name in names]

    missing = public - documented
    if missing:
        raise ValueError(
            "pyccolo_events: undocumented TraceEvent members "
            f"(add them to EVENTS): {sorted(missing)}"
        )
    extra = documented - public
    if extra:
        raise ValueError(
            "pyccolo_events: EVENTS entries with no matching TraceEvent "
            f"(stale, remove them): {sorted(extra)}"
        )
    dupes = [n for n in categorized if categorized.count(n) > 1]
    if dupes:
        raise ValueError(
            f"pyccolo_events: events listed in >1 category: {sorted(set(dupes))}"
        )
    uncategorized = public - set(categorized)
    if uncategorized:
        raise ValueError(
            "pyccolo_events: events missing from CATEGORIES: "
            f"{sorted(uncategorized)}"
        )


class PyccoloEventsDirective(Directive):
    has_content = False

    def run(self):
        _check_coverage()
        ast_hooks = _ast_hooks()

        lines: List[str] = []
        for title, names in CATEGORIES:
            lines.append(f".. rubric:: {title}")
            lines.append("")
            lines.append(".. list-table::")
            lines.append("   :header-rows: 1")
            lines.append("   :widths: 24 40 28 16")
            lines.append("")
            lines.append("   * - Event")
            lines.append("     - Fires on")
            lines.append("     - Handler return")
            lines.append("     - Flags")
            for name in names:
                fires_on, ret = EVENTS[name]
                flags = []
                evt = TraceEvent[name]
                if evt in BEFORE_EXPR_EVENTS:
                    flags.append("thunk")
                if evt in SYS_TRACE_EVENTS:
                    flags.append("sys")
                if name in ast_hooks:
                    flags.append(f"ast: ``{ast_hooks[name]}``")
                lines.append(f"   * - ``{name}``")
                lines.append(f"     - {fires_on}")
                lines.append(f"     - {ret}")
                lines.append(f"     - {', '.join(flags) if flags else '—'}")
            lines.append("")

        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(
            StringList(lines, source="<pyccolo-events>"), self.content_offset, node
        )
        return node.children


def setup(app):
    app.add_directive("pyccolo-events", PyccoloEventsDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
