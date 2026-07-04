Use sys.settrace-style events
=============================

**Goal:** hook Python's built-in tracing (frame pushes/pops, per-line, exceptions)
instead of AST-level events.

Pyccolo supports instrumentation via Python's `built-in tracing utilities
<https://docs.python.org/3/library/sys.html#sys.settrace>`_. Register a handler for
one of the corresponding events — ``call``, ``line``, ``return_``, ``exception``,
or ``opcode``:

.. code-block:: python

   import pyccolo as pyc

   class SysTracer(pyc.BaseTracer):
       @pyc.call
       def handle_call(self, *_, **__):
           print("Pushing a stack frame!")

       @pyc.return_
       def handle_return(self, *_, **__):
           print("Popping a stack frame!")

   with SysTracer:
       def f():
           def g():
               return 42
           return g()
       # push, push, pop, pop
       answer = f()

.. note::

   No ``pyc.exec`` is needed here — Python's built-in tracing does not involve any
   AST transformation, so code in the same file as the tracer is traced directly.
   (``opcode`` events require Python ≥ 3.7.)

Pyccolo also composes with an *existing* ``sys.settrace`` function: its ``call``
and ``return_`` tests pass even when `coverage.py
<https://coverage.readthedocs.io/>`_ is active — and without breaking it.

.. seealso::

   In the :doc:`/reference/events` table these events carry the **sys** flag.
