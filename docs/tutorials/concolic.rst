Tutorial: concolic execution
============================

*Concolic* (concrete + symbolic) execution runs a program on real inputs while
simultaneously building up the symbolic **path constraint** that describes why it
went the way it did. Negating one branch of that constraint and solving gives you
inputs that drive the program down a *different* path — so you can automatically
discover every branch. This tutorial follows the shipped ``concolic.py`` example
and drives its explorer to find the paths through a function.

The one thing operator overloading can't do
--------------------------------------------

Most of a symbolic executor can be built with ordinary Python: wrap each input in
a proxy object whose arithmetic and comparison operators build a symbolic term as
a side effect. The example calls this proxy ``Sym``:

.. code-block:: python

   class Sym:
       __slots__ = ("v", "t")           # concrete value, symbolic term

       def __init__(self, v, t):
           self.v, self.t = v, t

       def __mul__(self, other):
           ov, ot = _split(other)
           return Sym(self.v * ov, Bin("*", self.t, ot))   # x * 2  ->  Sym(.., (x * 2))
       # ... __add__, __lt__, __gt__, __eq__, and friends similarly ...

That handles *data* flow. What overloading **cannot** intercept is *control* flow:
when Python evaluates ``if x > 0:``, it calls ``bool()`` on the condition and then
branches — and ``bool()`` must return a real ``True``/``False``, discarding the
symbolic term. This is exactly the seam Pyccolo fills.

Capturing branch conditions
---------------------------

Pyccolo fires ``after_if_test`` and ``after_while_test`` with the *value the branch
is about to use*, and lets the handler replace it. The tracer records the symbolic
condition as a path constraint, then returns the plain concrete bool so CPython
takes the same branch the concrete run would:

.. code-block:: python

   class ConcolicTracer(pyc.BaseTracer):
       instrument_all_files = True       # instrument the target function's file too

       @pyc.register_raw_handler(pyc.after_if_test)
       def handle_if_test(self, ret, node_id, *_, **__):
           if not isinstance(ret, Sym):
               return ret                # concrete condition: nothing to record
           self.path.append(Branch(node_id, bool(ret.v), _as_bool_term(ret.t)))
           return bool(ret.v)            # keep CPython on the concrete path

``else`` needs no special handling — it is just the fall-through when the ``if``
records ``False`` — and Python desugars ``elif`` into a nested ``if``, so each one
gets its own event for free.

Short-circuit ``and`` / ``or`` are handled precisely with the **thunk** that
``before_boolop_arg`` provides (see :doc:`/guides/observe_and_override`): the
tracer wraps each operand's thunk so it records that operand's term *only if it is
actually evaluated*. On the path where ``x > 0`` is false, the ``y > 0`` operand of
``x > 0 and y > 0`` is never forced, so it never enters the constraint.

The search loop
---------------

After a run, the tracer holds an ordered list of branch decisions. To reach a new
path, the ``explore`` helper negates one decision, keeps the earlier ones, hands
the conjunction to a solver (Z3 if installed, otherwise a small bounded
brute-force search), and re-runs with the solved inputs — repeating until no new
paths appear. Note ``tracer.instrumented(func)``: the target function is
instrumented in place (see :doc:`/guides/tracing_real_programs`) so its branch
tests emit events.

Exploring a function
--------------------

Let's drive it. ``classify`` has four reachable paths, and concolic exploration
finds all of them starting from a single seed input:

.. testcode::

   from pyccolo.examples import concolic
   from pyccolo.examples.concolic import explore

   results = explore(concolic.classify, {"x": 0, "y": 0})
   assert {r.retval for r in results} == {
       "x<0", "x==0", "x>0, y<=x", "x>0, y>x",
   }

The real power shows up when a branch is only reachable for one precise input. In
``needs_solver``, ``y = x * 2`` and the code checks ``y == 12`` then ``x > 5`` —
so the ``"jackpot"`` branch is reachable *only* when ``x == 6``, and the solver
derives that for you:

.. testcode::

   results = explore(concolic.needs_solver, {"x": 0})
   jackpot = [r.inputs for r in results if r.retval == "jackpot"]
   assert jackpot == [{"x": 6}]        # solved from the path constraint, not guessed

Because ``y == 12`` forces ``x == 6`` and ``6 > 5`` always holds, the ``"almost"``
branch is *unreachable* — and concolic exploration simply never reports it.
Short-circuit branches work too: ``region`` uses ``and`` / ``or`` in its tests,
and each discovered path's constraint reflects exactly the operands that ran.

.. testcode::

   results = explore(concolic.region, {"x": 0, "y": 0})
   assert {r.retval for r in results} == {
       "has-negative", "on-axis", "quadrant-1",
   }

Run ``python pyccolo/examples/concolic.py`` from the repository root to see the
full demo, including the path constraint printed for each discovered path.

Where to next
-------------

Concolic execution leans on three Pyccolo features you've now seen: overriding a
branch test's value (:doc:`/guides/observe_and_override`), the short-circuit thunk
of ``before_boolop_arg``, and instrumenting an ordinary function
(:doc:`/guides/tracing_real_programs`). The
:doc:`watchpoint debugger </tutorials/watchpoint_debugger>` is a gentler next
build if the symbolic machinery here was a lot at once.
