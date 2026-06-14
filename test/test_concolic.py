# -*- coding: utf-8 -*-
"""
Tests for the concolic-execution example (``pyccolo/examples/concolic.py``).
"""
import types

from pyccolo.examples.concolic import (
    And,
    Bin,
    Cmp,
    Lit,
    Not,
    Or,
    Sym,
    Var,
    explore,
)


def _pristine(func):
    """A clean (un-instrumented) copy of ``func``.

    ``explore`` rebinds ``func.__code__`` to instrumented bytecode, but the
    original code object is left untouched -- snapshot it so we can use the
    function as an oracle.
    """
    return types.FunctionType(
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__,
    )


def _retvals(results):
    return {r.retval for r in results}


def _assert_inputs_reproduce(oracle, results):
    """Every discovered input, run concretely, must yield its recorded retval."""
    for r in results:
        assert oracle(**r.inputs) == r.retval, r


# ---------------------------------------------------------------------------
# Symbolic term semantics.
# ---------------------------------------------------------------------------
def test_term_eval():
    x, y = Var("x"), Var("y")
    assert Bin("+", x, Lit(1)).eval({"x": 4}) == 5
    assert Bin("*", x, Lit(2)).eval({"x": 6}) == 12
    assert Cmp(">", x, Lit(0)).eval({"x": 1}) is True
    assert Cmp(">", x, Lit(0)).eval({"x": -1}) is False
    assert Not(Cmp(">", x, Lit(0))).eval({"x": -1}) is True

    both_pos = And([Cmp(">", x, Lit(0)), Cmp(">", y, Lit(0))])
    assert both_pos.eval({"x": 1, "y": 1}) is True
    assert both_pos.eval({"x": 1, "y": -1}) is False

    any_neg = Or([Cmp("<", x, Lit(0)), Cmp("<", y, Lit(0))])
    assert any_neg.eval({"x": 5, "y": -1}) is True
    assert any_neg.eval({"x": 5, "y": 5}) is False


def test_sym_operator_overloading_builds_terms():
    x = Sym(3, Var("x"))
    expr = x * 2 + 1  # Sym arithmetic -> concrete 7, term ((x * 2) + 1)
    assert expr.v == 7
    assert str(expr.t) == "((x * 2) + 1)"

    eq = (x * 2) == 6  # 3*2 == 6 -> True
    assert eq.v is True
    assert str(eq.t) == "((x * 2) == 6)"

    rsub = 10 - x  # reflected operator -> 7, term (10 - x)
    assert rsub.v == 7
    assert str(rsub.t) == "(10 - x)"


# ---------------------------------------------------------------------------
# Path exploration.
# ---------------------------------------------------------------------------
def test_classify_full_coverage():
    def classify(x, y):
        if x > 0:
            if y > x:
                return "x>0, y>x"
            else:
                return "x>0, y<=x"
        elif x == 0:
            return "x==0"
        else:
            return "x<0"

    oracle = _pristine(classify)
    results = explore(classify, {"x": 0, "y": 0})

    assert _retvals(results) == {"x>0, y>x", "x>0, y<=x", "x==0", "x<0"}
    assert len(results) == 4
    _assert_inputs_reproduce(oracle, results)


def test_solver_inverts_arithmetic_and_proves_dead_branch():
    def needs_solver(x):
        y = x * 2
        if y == 12:
            if x > 5:
                return "jackpot"
            return "almost"  # unreachable: y==12 forces x==6 > 5
        return "nope"

    oracle = _pristine(needs_solver)
    results = explore(needs_solver, {"x": 0})

    assert _retvals(results) == {"nope", "jackpot"}
    assert "almost" not in _retvals(results)
    assert len(results) == 2
    _assert_inputs_reproduce(oracle, results)

    # The non-trivial input must have been solved for, not stumbled upon.
    jackpot = next(r for r in results if r.retval == "jackpot")
    assert jackpot.inputs["x"] == 6


# ---------------------------------------------------------------------------
# Short-circuit and / or via before_boolop_arg.
# ---------------------------------------------------------------------------
def test_and_short_circuit_omits_unevaluated_operand():
    def both_positive(x, y):
        if x > 0 and y > 0:
            return "both"
        return "not"

    oracle = _pristine(both_positive)
    results = explore(both_positive, {"x": 0, "y": 0})

    assert _retvals(results) == {"both", "not"}
    _assert_inputs_reproduce(oracle, results)

    # On the path where x > 0 is false, y > 0 is never evaluated, so its term
    # must not appear in the recorded constraint.
    short_circuited = [
        r
        for r in results
        if r.retval == "not" and all("y" not in str(c) for c in r.constraints)
    ]
    assert short_circuited, "expected a short-circuited path that never reads y"

    # And there must also be a path where both operands were evaluated.
    full = [r for r in results if any("y" in str(c) for c in r.constraints)]
    assert full, "expected a path where both operands were evaluated"


def test_or_short_circuit_and_coverage():
    def any_negative(x, y):
        if x < 0 or y < 0:
            return "neg"
        return "nonneg"

    oracle = _pristine(any_negative)
    results = explore(any_negative, {"x": 5, "y": 5})

    assert _retvals(results) == {"neg", "nonneg"}
    _assert_inputs_reproduce(oracle, results)

    # x < 0 true short-circuits the or, so y is not read on that path.
    short_circuited = [
        r
        for r in results
        if r.retval == "neg" and all("y" not in str(c) for c in r.constraints)
    ]
    assert short_circuited, "expected x<0 to short-circuit the or"


def test_region_elif_with_boolops():
    def region(x, y):
        if x > 0 and y > 0:
            return "quadrant-1"
        elif x < 0 or y < 0:
            return "has-negative"
        else:
            return "on-axis"

    oracle = _pristine(region)
    results = explore(region, {"x": 0, "y": 0})

    assert _retvals(results) == {"quadrant-1", "has-negative", "on-axis"}
    _assert_inputs_reproduce(oracle, results)


# ---------------------------------------------------------------------------
# Conditions with no symbolic dependence record no constraint.
# ---------------------------------------------------------------------------
def test_constant_condition_records_no_constraint():
    def const_cond(x):
        if 2 > 1:
            return "yes"
        return "no"

    oracle = _pristine(const_cond)
    results = explore(const_cond, {"x": 0})

    assert len(results) == 1
    assert results[0].retval == "yes"
    assert results[0].constraints == []
    _assert_inputs_reproduce(oracle, results)


def test_constant_boolop_records_no_constraint():
    def const_boolop(x):
        if 1 > 0 and 2 > 0:
            return "yes"
        return "no"

    results = explore(const_boolop, {"x": 0})
    assert len(results) == 1
    assert results[0].retval == "yes"
    assert results[0].constraints == []
