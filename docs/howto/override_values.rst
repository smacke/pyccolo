Override the value of an expression
===================================

**Goal:** change what a piece of instrumented code evaluates to, not just observe
it.

For value-carrying events, the value your handler returns **replaces** the value
of the instrumented expression. Returning ``None`` (or nothing) means *no
override*.

.. code-block:: python

   import pyccolo as pyc

   class IncrementEveryAssignment(pyc.BaseTracer):
       @pyc.after_assign_rhs
       def handle(self, ret, *_, **__):
           return ret + 1

   with IncrementEveryAssignment:
       env = pyc.exec("x = 42")
       assert env["x"] == 43

Because literal events can override the value flowing out of a literal, a whole
behavioral change can fit in one handler — here, making every ``float`` literal
exact by promoting it to :class:`~decimal.Decimal`:

.. code-block:: python

   from decimal import Decimal
   import pyccolo as pyc

   class ExactFloats(pyc.BaseTracer):
       @pyc.after_float
       def to_decimal(self, ret, *_, **__):
           return Decimal(str(ret))

   with ExactFloats:
       pyc.exec("print(0.1 + 0.2)")  # -> 0.3   (not 0.30000000000000004)

Overriding with ``None``
------------------------

Since returning ``None`` means "don't override", use the ``pyc.Null`` sentinel to
*actually* substitute a real ``None``:

.. code-block:: python

   class NullsOutAssignments(pyc.BaseTracer):
       @pyc.register_handler(pyc.after_assign_rhs)
       def handle_1(self, *_, **__):
           return pyc.Null

       @pyc.register_handler(pyc.after_assign_rhs)
       def handle_2(self, ret, *_, **__):
           assert ret is None  # the previous handler's Null became a real None
           return ret

   env = NullsOutAssignments.instance().exec("x = 42", local_env={})
   assert env["x"] is None

Stopping other handlers
-----------------------

Two more sentinels control the handler/tracer stack for the current event:

- ``pyc.Skip`` — stop running further handlers for this event **on this tracer**.
- ``pyc.SkipAll`` — abort the rest of the **tracer stack** for this event, so no
  lower tracer's handler runs.

.. seealso::

   :doc:`/reference/handlers` for the full return-value contract and the thunk
   convention for before-expression events.
