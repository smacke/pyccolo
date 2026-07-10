Observe and override values
===========================

A handler is not limited to watching your program run. For **value-carrying
events**, whatever your handler returns *replaces* the value flowing through the
instrumented expression — and returning nothing means "leave it alone." That one
rule is enough to build behavioral changes that would otherwise require rewriting
the program. This guide collects the moves that rule enables: overriding values,
the sentinels that give you finer control, gating a handler to just the nodes you
care about, and intercepting an expression *before* it runs.

Override a value by returning it
--------------------------------

Every ``after_*`` event that carries a value hands your handler that value as its
first argument (conventionally ``ret``) and uses your return value in its place.
Here we add one to the right-hand side of every assignment:

.. testcode::

   import pyccolo as pyc


   class IncrementEveryAssignment(pyc.BaseTracer):
       @pyc.after_assign_rhs
       def handle(self, ret, *_, **__):
           return ret + 1


   with IncrementEveryAssignment:
       env = pyc.exec("x = 42")
   assert env["x"] == 43

``pyc.exec`` returns the resulting namespace as a dict, which makes the effect
easy to inspect. The same idea works on literals — promoting every ``float`` to
an exact :class:`~decimal.Decimal`:

.. testcode::

   from decimal import Decimal


   class ExactFloats(pyc.BaseTracer):
       @pyc.after_float
       def to_decimal(self, ret, *_, **__):
           return Decimal(str(ret))


   with ExactFloats:
       assert pyc.exec("x = 0.1 + 0.2")["x"] == Decimal("0.3")

...or redacting digits out of every string literal with a regex — an override
that reaches into ordinary-looking code and quietly changes its data:

.. testcode::

   import re


   class Redact(pyc.BaseTracer):
       @pyc.after_string
       def scrub(self, ret, *_, **__):
           return re.sub(r"\d", "*", ret)


   with Redact:
       assert pyc.exec("s = 'call 5551234'")["s"] == "call *******"

Returning ``None`` means "don't override"
------------------------------------------

There is one wrinkle: since a handler that returns ``None`` (or falls off the
end) means *no override*, you cannot substitute a real ``None`` just by returning
it. Use the :data:`pyc.Null` sentinel to override *with* ``None``:

.. testcode::

   class NullOutAssignments(pyc.BaseTracer):
       @pyc.after_assign_rhs
       def handle(self, *_, **__):
           return pyc.Null


   with NullOutAssignments:
       assert pyc.exec("x = 42")["x"] is None

The four sentinels
------------------

``pyc.Null`` is one of a small family of sentinels a handler can return to
control the flow of instrumentation. The others govern the *handler stack* — when
more than one handler (on one tracer, or across composed tracers) is registered
for the same event, they run in order and each receives the previous one's result
as ``ret``.

- :data:`pyc.Null` — override the value with a real ``None``.
- :data:`pyc.Pass` — for ``before_stmt``, skip the statement entirely (it never
  runs).
- :data:`pyc.Skip` — stop running *this tracer's* remaining handlers for this
  event.
- :data:`pyc.SkipAll` — stop the *entire tracer stack* for this event, so no
  lower tracer's handler runs either.

``pyc.Pass`` turns ``before_stmt`` into a veto. Here we drop every top-level
``print`` call without running it:

.. testcode::

   class SuppressPrints(pyc.BaseTracer):
       @pyc.before_stmt(
           when=pyc.Predicate(
               lambda node: isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call)
               and isinstance(node.value.func, ast.Name)
               and node.value.func.id == "print",
               static=True,
           )
       )
       def veto(self, *_, **__):
           return pyc.Pass


   with SuppressPrints:
       env = pyc.exec("print('never shown')\nkept = 5")
   assert env["kept"] == 5

And ``pyc.Skip`` halts the rest of a tracer's handler chain for the current
event:

.. testcode::

   ran = []


   class StopsEarly(pyc.BaseTracer):
       @pyc.register_handler(pyc.after_assign_rhs)
       def first(self, ret, *_, **__):
           ran.append("first")
           return pyc.Skip

       @pyc.register_handler(pyc.after_assign_rhs)
       def second(self, ret, *_, **__):
           ran.append("second")
           return ret


   with StopsEarly:
       pyc.exec("x = 1")
   assert ran == ["first"]

See :doc:`/reference/handlers` for the full return-value contract.

.. _when-predicate-guide:

Fire a handler only sometimes
-----------------------------

Pay-as-you-go decides whether an event is emitted *at all*; a ``when=`` predicate
goes finer and gates an *individual* handler to only the nodes it cares about. In
its simplest form it is a plain callable of the AST node:

.. testcode::

   seen = []


   class OnlyAdditions(pyc.BaseTracer):
       @pyc.after_binop(when=lambda node: isinstance(node.op, ast.Add))
       def handle(self, ret, node, *_, **__):
           seen.append(type(node.op).__name__)
           return ret


   with OnlyAdditions:
       pyc.exec("a = 1 + 2\nb = 3 * 4")
   assert seen == ["Add"]      # the multiplication never reached the handler

To combine conditions, just combine them in the callable:

.. testcode::

   class AddOrSub(pyc.BaseTracer):
       @pyc.after_binop(when=lambda node: isinstance(node.op, (ast.Add, ast.Sub)))
       def handle(self, ret, *_, **__):
           return ret


   with AddOrSub:
       pyc.exec("a = 1 + 2 - 3")

Static vs. dynamic predicates
-----------------------------

Wrapping the condition in a :class:`pyccolo.Predicate` lets you say *when* it is
evaluated:

- ``pyc.Predicate(cond, static=True)`` is evaluated **once, at instrument time**,
  and the answer is baked into the rewrite — zero runtime cost, but the condition
  may only look at the shape of the syntax tree. This is what the ``SuppressPrints``
  example above uses: whether a statement is a ``print`` call is decidable from
  the AST alone.
- ``pyc.Predicate(cond)`` (dynamic) re-checks on every event.

Because a static predicate is resolved during the rewrite, a handler that never
matches adds *nothing* to your running program — the same spirit as pay-as-you-go
itself. Reach for ``static=True`` whenever the decision depends only on syntax.

Register by AST node type
-------------------------

Instead of naming an event, you can register a handler by the :mod:`ast` node
type it should fire for; Pyccolo maps the node to the corresponding value-carrying
event. ``ast.Assign`` maps to ``after_assign_rhs``, ``ast.BinOp`` to
``after_binop``, ``ast.Call`` to ``after_call``, and so on:

.. testcode::

   class Bump(pyc.BaseTracer):
       @pyc.register_handler(ast.Assign)
       def handle(self, ret, *_, **__):
           return ret + 100


   with Bump:
       assert pyc.exec("x = 2 * 3")["x"] == 106

The full node-to-event mapping lives in :doc:`/reference/registration`.

Intercept an expression before it runs
---------------------------------------

Overriding an ``after_*`` value happens *after* the expression has been
evaluated. Sometimes you need to decide whether it runs at all — and the
``before_*`` events give you that, by passing your handler a **thunk**: a
zero-argument callable that, when invoked, evaluates the original expression.
Return a thunk of your own and Pyccolo calls it in place of the original.

This is exactly what short-circuit evaluation needs. ``before_boolop_arg`` fires
once per operand of an ``and`` / ``or``, so we can record which operands actually
get evaluated:

.. testcode::

   evaluated = []


   class TraceShortCircuit(pyc.BaseTracer):
       @pyc.register_handler(pyc.before_boolop_arg)
       def handle(self, thunk, *_, **__):
           def wrapped():
               value = thunk()          # evaluate the operand for real...
               evaluated.append(value)  # ...and record what it produced
               return value

           return wrapped


   with TraceShortCircuit:
       assert pyc.exec("r = False and (1 / 0)")["r"] is False
   assert evaluated == [False]          # the `1 / 0` operand was never evaluated

Because ``and`` short-circuits, the second operand's thunk is never called — so
the ``1 / 0`` that would have raised is quietly skipped, and it never reaches our
list. The reference marks which events pass a thunk (the "before-expression"
events) in the :doc:`event catalog </reference/events>`.
