Handlers
========

A *handler* is a method on your tracer subclass that Pyccolo calls when an
:doc:`event <events>` fires. This page is the precise contract: what a handler
receives, what its return value means, and the sentinels it can return.

The signature
-------------

Every handler is called with four positional arguments, followed by
event-specific keyword arguments:

.. code-block:: python

   def handle(self, ret, node, frame, event, *, guard=None, **kwargs):
       ...

============  ================================================================
Parameter     Meaning
============  ================================================================
``ret``       The value currently flowing through the instrumented expression
              (the previous handler's result, or the runtime value). For
              *before-expression* events this is a **thunk** — a zero-argument
              callable that produces the value (see below).
``node``      The :class:`ast.AST` node being instrumented — or its integer id
              when the handler was registered "raw" (see
              :func:`~pyccolo.register_raw_handler`), or ``None`` for ``sys``
              events.
``frame``     The :class:`~types.FrameType` executing at the instrumentation
              point.
``event``     The :class:`~pyccolo.trace_events.TraceEvent` that fired — useful
              when one handler is registered for several events.
============  ================================================================

Because those four are enough for most handlers, the idiomatic shape swallows
everything it does not need::

   @pyc.after_assign_rhs
   def handle(self, ret, *_, **__):
       return ret + 1

Keyword arguments
-----------------

Some events pass extra keyword arguments. The most commonly used:

- ``guard`` — the guard name associated with this handler at this node (or
  ``None``). Pass it to :meth:`~pyccolo.BaseTracer.activate_guard` to switch off
  further instrumentation of the enclosing function/loop; see
  :doc:`/guides/guards_and_performance`.
- ``ret_expr`` — for some statement events, the value of the trailing
  expression.
- ``is_last`` — for argument/operand sequences (e.g. ``before_boolop_arg``),
  whether this is the final element.
- ``attr_or_subscript`` — the attribute name or subscript key, for attribute /
  subscript events.

Always accept ``**kwargs`` (spelled ``**__`` by convention) so your handler keeps
working as events grow new keywords.

Return values
-------------

For value-carrying events, **the value a handler returns replaces the value of
the instrumented expression.** This is what lets a handler change behavior, not
merely observe it. Returning ``None`` (or falling off the end of the function)
means *no override*. When multiple handlers fire for one event — on the same
tracer, or across a :doc:`stack of tracers </concepts/composition>` — the return
value of each is threaded into the ``ret`` of the next, so overrides compose.

To express intents a bare ``return`` cannot, return one of the sentinels:

.. data:: pyccolo.Null

   Override the instrumented value **with** ``None``. (Returning ``None`` itself
   means "don't override", so this sentinel is how you deliberately substitute a
   real ``None``.)

.. data:: pyccolo.Skip

   Stop running any further handlers for this event **on the current tracer**.

.. data:: pyccolo.SkipAll

   Abort the rest of the tracer **stack** for this event — no lower tracer's
   handler for this event runs.

.. data:: pyccolo.Pass

   An explicit "do nothing / no override" marker. With ``before_stmt`` (and
   :meth:`~pyccolo.BaseTracer.exec_saved_thunk`), returning ``Pass`` as the saved
   thunk causes the statement to be skipped entirely.

The thunk convention (before-expression events)
-----------------------------------------------

Events flagged **thunk** in the :doc:`taxonomy <events>` (the members of
``BEFORE_EXPR_EVENTS``) fire *before* their expression is evaluated. For these,
``ret`` is a **zero-argument callable** that, when called, produces the value.
This lets a handler decide *whether* to evaluate the expression at all, evaluate
it more than once, or wrap it:

.. code-block:: python

   @pyc.before_binop
   def handle(self, ret_thunk, *_, **__):
       # ret_thunk() would compute `x + y`; return a callable to override.
       return lambda: 0

If you return a plain (non-callable) value from a before-expression handler,
Pyccolo wraps it into ``lambda *_: value`` for you.

Example
-------

Promoting every ``float`` literal to :class:`~decimal.Decimal` — a whole
behavioral change in one handler:

.. code-block:: python

   from decimal import Decimal
   import pyccolo as pyc

   class ExactFloats(pyc.BaseTracer):
       @pyc.after_float
       def to_decimal(self, ret, *_, **__):
           return Decimal(str(ret))

   with ExactFloats:
       pyc.exec("print(0.1 + 0.2)")  # -> 0.3
