# -*- coding: utf-8 -*-
"""
Concolic (concrete + symbolic) execution implemented with Pyccolo.

The idea:

  * Symbolic inputs are ordinary :class:`Sym` proxies that carry both a
    *concrete* value and a *symbolic term*.  Arithmetic and comparison
    operators are overloaded so that the symbolic term is built up
    automatically as the program computes -- e.g. ``x * 2`` where ``x`` is
    symbolic produces a new ``Sym`` whose term is ``(x * 2)``.

  * Pyccolo supplies the one thing operator overloading cannot: it intercepts
    each ``if`` / ``elif`` / ``while`` *test* (via ``after_if_test`` /
    ``after_while_test``) and lets the handler *return the value the branch
    actually uses*.  We return the plain concrete bool -- so CPython stays on
    the path the concrete run would take -- while recording the symbolic branch
    condition as a path constraint.

  * After a run we have an ordered list of branch decisions.  To reach a new
    path we negate one decision, keep the earlier ones, hand the conjunction to
    a solver (Z3 if installed, otherwise a small brute-force search), and re-run
    with the solved inputs.  Repeating this explores the program's paths.

Note that ``else`` needs no special handling: it is simply the fall-through when
the ``if`` test returns ``False``, so recording ``not(cond)`` at the ``if``
*is* the else-branch constraint.  ``elif`` is desugared by Python into a nested
``if`` inside ``orelse``, so each ``elif`` test gets its own event for free.

Run as ``python pyccolo/examples/concolic.py`` from the repository root.

Short-circuit ``and`` / ``or`` are handled precisely: ``before_boolop_arg``
hands us a *thunk* per operand (pyccolo defers each operand so the connective
can short-circuit), so we record an operand's symbolic term only when it is
actually evaluated.  The conjunction/disjunction of the evaluated operands then
becomes the condition's term -- e.g. ``x > 0 and y > 0`` yields ``((x > 0) &
(y > 0))`` when both run, but just ``(x > 0)`` on the path where ``x > 0`` is
false and ``y > 0`` is never reached.

Caveats (shared by every symbolic executor, called out here for honesty):
  * Chained comparisons (``0 < x < 10``) still desugar to a short-circuit
    ``and`` over ``bool()`` of each comparison; Python hands the connective the
    operand objects, so this works, but other ``__bool__`` coercions outside
    ``if`` / ``while`` / ``and`` / ``or`` are not tracked.
  * Symbolic values that cross into C builtins / extensions (``len``, ``re``,
    numpy, ...) decay to concrete; you would need models for those.
  * Ternary ``a if c else b`` is an ``ast.IfExp`` (a different node) and is not
    instrumented here -- only statement-level ``if`` / ``elif`` / ``while``.
"""
import ast
import inspect
import logging
from itertools import product
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import pyccolo as pyc

logger = logging.getLogger(__name__)

try:
    import z3  # type: ignore

    _HAVE_Z3 = True
except ImportError:  # pragma: no cover - depends on environment
    _HAVE_Z3 = False


# ---------------------------------------------------------------------------
# Symbolic terms: a tiny expression tree we can print, evaluate, and translate
# to Z3.  Each term knows how to ``eval`` itself against a concrete environment
# (used by the brute-force solver) and how to lower itself to Z3.
# ---------------------------------------------------------------------------
class Term:
    def eval(self, env: Dict[str, int]) -> Any:
        raise NotImplementedError

    def z3(self, zvars: Dict[str, Any]) -> Any:
        raise NotImplementedError


class Var(Term):
    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, env):
        return env[self.name]

    def z3(self, zvars):
        return zvars[self.name]

    def __str__(self):
        return self.name


class Lit(Term):
    def __init__(self, value: Any) -> None:
        self.value = value

    def eval(self, env):
        return self.value

    def z3(self, zvars):
        return self.value

    def __str__(self):
        return repr(self.value)


_BIN = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "//": lambda a, b: a // b,
    "%": lambda a, b: a % b,
}

_CMP = {
    "<": lambda a, b: a < b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    ">=": lambda a, b: a >= b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


class Bin(Term):
    def __init__(self, op: str, l: Term, r: Term) -> None:
        self.op, self.l, self.r = op, l, r

    def eval(self, env):
        return _BIN[self.op](self.l.eval(env), self.r.eval(env))

    def z3(self, zvars):
        l, r = self.l.z3(zvars), self.r.z3(zvars)
        # z3's ArithRef defines ``/`` (integer division on Int sort) and ``%``
        # but not ``__floordiv__``, so route "//" through ``/`` explicitly.
        if self.op == "//":
            return l / r
        return _BIN[self.op](l, r)

    def __str__(self):
        return f"({self.l} {self.op} {self.r})"


class Cmp(Term):
    def __init__(self, op: str, l: Term, r: Term) -> None:
        self.op, self.l, self.r = op, l, r

    def eval(self, env):
        return _CMP[self.op](self.l.eval(env), self.r.eval(env))

    def z3(self, zvars):
        return _CMP[self.op](self.l.z3(zvars), self.r.z3(zvars))

    def __str__(self):
        return f"({self.l} {self.op} {self.r})"


class Not(Term):
    def __init__(self, t: Term) -> None:
        self.t = t

    def eval(self, env):
        return not self.t.eval(env)

    def z3(self, zvars):
        return z3.Not(self.t.z3(zvars))

    def __str__(self):
        return f"!{self.t}"


class And(Term):
    def __init__(self, args: List[Term]) -> None:
        self.args = args

    def eval(self, env):
        return all(a.eval(env) for a in self.args)

    def z3(self, zvars):
        return z3.And(*[a.z3(zvars) for a in self.args])

    def __str__(self):
        return "(" + " & ".join(str(a) for a in self.args) + ")"


class Or(Term):
    def __init__(self, args: List[Term]) -> None:
        self.args = args

    def eval(self, env):
        return any(a.eval(env) for a in self.args)

    def z3(self, zvars):
        return z3.Or(*[a.z3(zvars) for a in self.args])

    def __str__(self):
        return "(" + " | ".join(str(a) for a in self.args) + ")"


# Terms that already denote a boolean (no truthiness coercion needed).
_BOOL_TERMS = (Cmp, Not, And, Or)


def _as_bool_term(t: Term) -> Term:
    """Coerce a term to a boolean one (``x`` -> ``x != 0``) for use in and/or."""
    return t if isinstance(t, _BOOL_TERMS) else Cmp("!=", t, Lit(0))


def _term_of(value: Any) -> Term:
    return value.t if isinstance(value, Sym) else Lit(value)


def _has_var(t: Term) -> bool:
    if isinstance(t, Var):
        return True
    if isinstance(t, (Bin, Cmp)):
        return _has_var(t.l) or _has_var(t.r)
    if isinstance(t, Not):
        return _has_var(t.t)
    if isinstance(t, (And, Or)):
        return any(_has_var(a) for a in t.args)
    return False  # Lit


# ---------------------------------------------------------------------------
# The symbolic value proxy.  Operator overloading propagates symbolic terms
# through ordinary Python expressions; ``__bool__`` returns the concrete truth
# value so the proxy behaves sanely if it leaks into a non-branch context.
# ---------------------------------------------------------------------------
def _split(other: Any) -> Tuple[Any, Term]:
    """Return ``(concrete, term)`` for either a ``Sym`` or a plain value."""
    if isinstance(other, Sym):
        return other.v, other.t
    return other, Lit(other)


class Sym:
    __slots__ = ("v", "t")

    def __init__(self, v: Any, t: Term) -> None:
        self.v = v  # concrete value
        self.t = t  # symbolic term

    def _bin(self, other, op, swap=False):
        ov, ot = _split(other)
        l, r = (ot, self.t) if swap else (self.t, ot)
        lv, rv = (ov, self.v) if swap else (self.v, ov)
        return Sym(_BIN[op](lv, rv), Bin(op, l, r))

    def __add__(self, o):
        return self._bin(o, "+")

    def __radd__(self, o):
        return self._bin(o, "+", swap=True)

    def __sub__(self, o):
        return self._bin(o, "-")

    def __rsub__(self, o):
        return self._bin(o, "-", swap=True)

    def __mul__(self, o):
        return self._bin(o, "*")

    def __rmul__(self, o):
        return self._bin(o, "*", swap=True)

    def __floordiv__(self, o):
        return self._bin(o, "//")

    def __mod__(self, o):
        return self._bin(o, "%")

    def __neg__(self):
        return Sym(-self.v, Bin("-", Lit(0), self.t))

    def _cmp(self, other, op):
        ov, ot = _split(other)
        return Sym(_CMP[op](self.v, ov), Cmp(op, self.t, ot))

    def __lt__(self, o):
        return self._cmp(o, "<")

    def __le__(self, o):
        return self._cmp(o, "<=")

    def __gt__(self, o):
        return self._cmp(o, ">")

    def __ge__(self, o):
        return self._cmp(o, ">=")

    def __eq__(self, o):
        return self._cmp(o, "==")

    def __ne__(self, o):
        return self._cmp(o, "!=")

    def __bool__(self):
        return bool(self.v)

    def __repr__(self):
        return f"Sym(v={self.v!r}, t={self.t})"


# ---------------------------------------------------------------------------
# The tracer: record one path constraint per branch, follow the concrete path.
# ---------------------------------------------------------------------------
class Branch(NamedTuple):
    node_id: int
    took: bool  # True if the then-branch (truthy test) was taken
    cond: Term  # the branch condition, normalized to a boolean term


class ConcolicTracer(pyc.BaseTracer):
    # Instrument whichever file the target function lives in, not just this one.
    instrument_all_files = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.path: List[Branch] = []
        # Stack of in-flight boolop frames; each frame collects the terms of the
        # operands actually evaluated (short-circuiting skips the rest).
        self.boolop_stack: List[List[Term]] = []

    def _record_branch(self, ret, node_id):
        if not isinstance(ret, Sym):
            # Concrete condition: nothing symbolic to record; don't override.
            return ret
        cond = _as_bool_term(ret.t)
        self.path.append(Branch(node_id, bool(ret.v), cond))
        # Return the plain concrete bool so CPython branches identically.
        return bool(ret.v)

    @pyc.register_raw_handler(pyc.after_if_test)
    def handle_if_test(self, ret, node_id, *_, **__):
        return self._record_branch(ret, node_id)

    @pyc.register_raw_handler(pyc.after_while_test)
    def handle_while_test(self, ret, node_id, *_, **__):
        return self._record_branch(ret, node_id)

    @pyc.register_raw_handler(pyc.before_boolop)
    def handle_before_boolop(self, _thunk, *_, **__):
        # Open a frame for this boolop. Returning None keeps pyccolo's deferred
        # thunk intact so the operands still evaluate (and short-circuit).
        self.boolop_stack.append([])

    @pyc.register_raw_handler(pyc.before_boolop_arg)
    def handle_before_boolop_arg(self, thunk, *_, **__):
        # ``thunk`` evaluates this operand on demand; wrap it so we record the
        # operand's term iff it is actually forced (i.e. not short-circuited).
        frame = self.boolop_stack[-1]

        def wrapped():
            value = thunk()
            frame.append(_term_of(value))
            return value

        return wrapped

    @pyc.register_handler(pyc.after_boolop)
    def handle_after_boolop(self, ret, node: ast.BoolOp, *_, **__):
        operand_terms = [_as_bool_term(t) for t in self.boolop_stack.pop()]
        if len(operand_terms) == 1:
            term = operand_terms[0]
        elif isinstance(node.op, ast.And):
            term = And(operand_terms)
        else:
            term = Or(operand_terms)
        concrete = ret.v if isinstance(ret, Sym) else ret
        # Only stay symbolic if some input flows in; otherwise hand back the
        # plain concrete result so a constant condition records no constraint.
        return Sym(concrete, term) if _has_var(term) else concrete


# ---------------------------------------------------------------------------
# Solving and the concolic search loop.
# ---------------------------------------------------------------------------
_BRUTE_BOUND = 64


def _constraint(branch: Branch, want_took: bool) -> Term:
    """The boolean term asserting ``branch`` goes in the ``want_took`` direction."""
    return branch.cond if want_took else Not(branch.cond)


def _solve(constraints: List[Term], varnames: List[str]) -> Optional[Dict[str, int]]:
    if _HAVE_Z3:
        zvars = {n: z3.Int(n) for n in varnames}
        solver = z3.Solver()
        for c in constraints:
            solver.add(c.z3(zvars))
        if solver.check() != z3.sat:
            return None
        model = solver.model()
        return {
            n: (model[zvars[n]].as_long() if model[zvars[n]] is not None else 0)
            for n in varnames
        }
    # Fallback: bounded brute-force search over the integer grid.
    rng = range(-_BRUTE_BOUND, _BRUTE_BOUND + 1)
    for combo in product(rng, repeat=len(varnames)):
        env = dict(zip(varnames, combo))
        try:
            if all(c.eval(env) for c in constraints):
                return env
        except (ZeroDivisionError, ValueError):
            continue
    return None


def _key(inputs: Dict[str, int]) -> Tuple:
    return tuple(sorted(inputs.items()))


class Result(NamedTuple):
    inputs: Dict[str, int]
    decisions: Tuple[Tuple[int, bool, str], ...]
    constraints: List[Term]
    retval: Any


def explore(func, initial_inputs: Dict[str, int], max_runs: int = 64) -> List[Result]:
    """Concolically explore ``func``, returning one :class:`Result` per path."""
    varnames = list(inspect.signature(func).parameters)
    tracer = ConcolicTracer.instance()
    instrumented = tracer.instrumented(func)

    results: List[Result] = []
    seen_decisions = set()
    queue: List[Dict[str, int]] = [dict(initial_inputs)]
    queued = {_key(queue[0])}

    runs = 0
    while queue and runs < max_runs:
        inputs = queue.pop(0)
        runs += 1

        args = [Sym(inputs[name], Var(name)) for name in varnames]
        tracer.path = []
        tracer.boolop_stack = []
        try:
            retval = instrumented(*args)
        except Exception as exc:  # a path that raises is still a path
            retval = exc
        path = tracer.path

        # Identify a path by its branch outcomes *and* condition structure:
        # short-circuiting can make two runs agree on every ``if`` outcome yet
        # evaluate different operands (e.g. ``x>0`` alone vs ``x>0 and y>0``).
        decisions = tuple((b.node_id, b.took, str(b.cond)) for b in path)
        if decisions in seen_decisions:
            continue
        seen_decisions.add(decisions)
        results.append(
            Result(
                inputs=inputs,
                decisions=decisions,
                constraints=[_constraint(b, b.took) for b in path],
                retval=retval.v if isinstance(retval, Sym) else retval,
            )
        )

        # Generate children: flip each branch in turn, keeping the prefix.
        for i, branch in enumerate(path):
            constraints = [_constraint(path[j], path[j].took) for j in range(i)]
            constraints.append(_constraint(branch, not branch.took))
            model = _solve(constraints, varnames)
            if model is None:
                continue
            child = {name: int(model.get(name, 0)) for name in varnames}
            if _key(child) not in queued:
                queued.add(_key(child))
                queue.append(child)

    return results


# ---------------------------------------------------------------------------
# Demo.
# ---------------------------------------------------------------------------
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


def needs_solver(x):
    # Reachable only when x == 6: y == 12 forces x to 6, then x > 5 holds.
    y = x * 2
    if y == 12:
        if x > 5:
            return "jackpot"
        return "almost"
    return "nope"


def region(x, y):
    # Short-circuit and/or: the recorded path constraint reflects exactly the
    # operands that were evaluated on each concrete run.
    if x > 0 and y > 0:
        return "quadrant-1"
    elif x < 0 or y < 0:
        return "has-negative"
    else:
        return "on-axis"


def _run_demo(func, initial):
    sig = ", ".join(inspect.signature(func).parameters)
    logger.info("=== %s(%s) ===", func.__name__, sig)
    results = explore(func, initial)
    for r in results:
        call = ", ".join(f"{k}={v}" for k, v in r.inputs.items())
        path = " & ".join(str(c) for c in r.constraints) or "(no branches)"
        logger.info("  %s(%s) -> %r", func.__name__, call, r.retval)
        logger.info("      path: %s", path)
    logger.info("  discovered %d path(s)", len(results))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info("solver backend: %s", "z3" if _HAVE_Z3 else "brute-force")
    _run_demo(classify, {"x": 0, "y": 0})
    _run_demo(needs_solver, {"x": 0})
    _run_demo(region, {"x": 0, "y": 0})
