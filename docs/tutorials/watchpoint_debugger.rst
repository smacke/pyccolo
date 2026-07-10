Tutorial: a variable watchpoint debugger
========================================

Debuggers let you set a *watchpoint*: "stop (or tell me) whenever this variable
changes." In this tutorial you'll build one in about a dozen lines — a tracer that
fires a callback every time a chosen name is reassigned, handing you the old and
new values. It's a compact, practical example of *observing* execution (no value
overriding), reading the AST node and the live frame together, and threading your
own state through a tracer.

Unlike the syntax examples, this needs no new syntax and no Python-version floor —
just the assignment event.

Step 1: catch every assignment
------------------------------

The ``after_assign_rhs`` event fires with the value a right-hand side just
produced. That value is *about* to be bound to the target(s) on the left, so it's
the perfect moment to notice a change. A handler receives the AST ``node`` and the
live ``frame``; from the node we can find the enclosing assignment statement, and
from the frame we can read what the variable held a moment ago:

.. testcode::

   import ast
   import pyccolo as pyc


   class Watchpoint(pyc.BaseTracer):
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.callbacks = {}                     # name -> callback(name, old, new)

       def watch(self, name, callback):
           self.callbacks[name] = callback
           return self

       @pyc.after_assign_rhs
       def on_assign(self, ret, node, frame, *_, **__):
           stmt = self.containing_stmt_by_id.get(id(node))
           if isinstance(stmt, ast.Assign):
               for target in stmt.targets:
                   if isinstance(target, ast.Name) and target.id in self.callbacks:
                       old = frame.f_locals.get(target.id, "<unset>")
                       self.callbacks[target.id](target.id, old, ret)
           return ret            # observe only — never change the value

Two Pyccolo details do the work. ``self.containing_stmt_by_id`` is a bookkeeping
table Pyccolo maintains during the rewrite; it takes us from the right-hand-side
expression node up to the ``ast.Assign`` that contains it, so we can read the
assignment's *targets*. And ``frame.f_locals`` still holds the variable's
*previous* value, because the new one has not been bound yet — that's what makes
"old → new" possible. Returning ``ret`` unchanged keeps this a pure observer.

Step 2: set a watch and run
---------------------------

Register a callback for the name you care about, then run some code under the
tracer. Here we record every change to ``balance`` into a list:

.. testcode::

   changes = []

   tracer = Watchpoint.instance().watch(
       "balance", lambda name, old, new: changes.append((old, new))
   )

   with tracer:
       pyc.exec(
           "balance = 100\n"
           "balance = balance - 30\n"
           "other = 5\n"            # not watched -> ignored
           "balance = balance + 10\n"
       )

   assert changes == [("<unset>", 100), (100, 70), (70, 80)]

The unwatched assignment to ``other`` is skipped, and each write to ``balance``
arrives with the value it had just before. A real debugger would print or drop
into a prompt instead of appending to a list — that is just a different callback:

.. code-block:: python

   Watchpoint.instance().watch(
       "balance",
       lambda name, old, new: print(f"{name}: {old!r} -> {new!r}"),
   )
   # balance: '<unset>' -> 100
   # balance: 100 -> 70
   # ...

Step 3: watch by condition
--------------------------

Because the callback is ordinary Python, "break only when the value goes negative"
or "only on the third change" is just a conditional inside it:

.. code-block:: python

   def alert_if_negative(name, old, new):
       if isinstance(new, (int, float)) and new < 0:
           raise AssertionError(f"{name} went negative: {new}")

   Watchpoint.instance().watch("balance", alert_if_negative)

For a version that only inspects — rather than reads the frame's previous value —
you could subscribe to ``load_name`` instead and watch *reads*, or combine both to
log the entire lifetime of a variable.

Where to next
-------------

You've now used the two halves of a handler's context — the static AST (via
``containing_stmt_by_id``) and the dynamic frame — without overriding anything.
The :doc:`/guides/observe_and_override` guide covers the flip side (changing values
in flight), and :doc:`/guides/guards_and_performance` shows how to keep a
per-variable watch cheap with local guards once you only need the first change.
