The model: events and handlers
==============================

Pyccolo exposes a fine-grained taxonomy of over 100 events. A *tracer* is a
subclass of :class:`pyccolo.BaseTracer` that registers *handlers* for the events
it cares about; Pyccolo rewrites source so that each event fires at exactly the
right point during execution.

Common events
-------------

The full list lives in :class:`pyccolo.trace_events.TraceEvent` (and every
member is re-exported as ``pyc.<event>``). Some of the more common ones:

- ``pyc.before_stmt`` / ``pyc.after_stmt``, emitted around statements;
- ``pyc.before_attribute_load`` / ``pyc.after_attribute_load``, emitted in
  `load contexts <https://docs.python.org/3/library/ast.html#ast.Load>`_ around
  attribute accesses;
- ``pyc.load_name``, emitted when a variable is used in a load context (e.g.
  ``foo`` in ``bar = foo.baz``);
- ``pyc.before_binop`` / ``pyc.after_binop``, ``pyc.before_unaryop`` /
  ``pyc.after_unaryop``, emitted around binary (e.g. ``x + y``) and unary (e.g.
  ``-x``, ``not x``) operations;
- ``pyc.after_assign_rhs``, emitted after the right-hand side of an assignment;
- literal events like ``pyc.after_int`` / ``pyc.after_float`` /
  ``pyc.after_string``;
- ``pyc.call`` and ``pyc.return_``, two non-AST trace events built in to Python
  (see :doc:`sys_settrace`).

The handler signature
---------------------

Every handler is passed four positional arguments:

1. the **return value**, for instrumented expressions;
2. the **AST node** (or node id, if using :func:`pyccolo.register_raw_handler`,
   or ``None`` for ``sys`` events);
3. the **stack frame**, at the point where instrumentation kicks in;
4. the **event** (useful when the same handler is registered for multiple
   events).

Some events pass additional keyword arguments, but the four above suffice for
most use cases — hence the ubiquitous ``def handle(self, ret, *_, **__)`` shape,
which binds the return value and ignores the rest.

Handlers override, not just observe
-----------------------------------

For many events, the value a handler returns *replaces* the value of the
instrumented expression:

.. code-block:: python

   import pyccolo as pyc


   class IncrementEveryAssignment(pyc.BaseTracer):
       @pyc.after_assign_rhs
       def handle(self, ret, *_, **__):
           return ret + 1


   with IncrementEveryAssignment:
       env = pyc.exec("x = 42")
       assert env["x"] == 43

Returning ``None`` (or nothing) means "don't override." The return-value
protocol has a few sentinels for the cases plain ``None`` can't express:

- ``pyccolo.Null`` — override the value *with* ``None`` (as opposed to "no
  override");
- ``pyccolo.Skip`` — stop running further handlers for the current event on
  this tracer;
- ``pyccolo.SkipAll`` — abort the whole tracer stack for the current event.

A whole behavioral change in one handler
----------------------------------------

Because literal events can override the value that flows out of a literal, an
entire behavioral change can fit in a handler. Here is a tracer that makes every
float literal *exact* by promoting it to :class:`~decimal.Decimal`:

.. code-block:: python

   from decimal import Decimal

   import pyccolo as pyc


   class ExactFloats(pyc.BaseTracer):
       @pyc.after_float
       def to_decimal(self, ret, *_, **__):
           return Decimal(str(ret))


   with ExactFloats:
       pyc.exec("print(0.1 + 0.2)")  # -> 0.3   (not 0.30000000000000004)

Instrumentation is pay-as-you-go
--------------------------------

For AST events, Python source is only transformed to emit an event when there is
at least one active tracer with at least one handler registered for that event.
This keeps the transformed source from becoming bloated when only a few events
are needed.
