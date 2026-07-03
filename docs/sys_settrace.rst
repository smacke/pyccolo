Compatibility with ``sys.settrace``
===================================

Pyccolo supports not only AST-level instrumentation, but also instrumentation
built on Python's `built-in tracing utilities
<https://docs.python.org/3/library/sys.html#sys.settrace>`_. To use it, register
handlers for one of the corresponding Pyccolo events: ``call``, ``line``,
``return_``, ``exception``, or ``opcode``.

.. code-block:: python

   import pyccolo as pyc


   class SysTracer(pyc.BaseTracer):
       @pyc.call
       def handle_call(self, *_, **__):
           print("Pushing a stack frame!")

       @pyc.return_
       def handle_return(self, *_, **__):
           print("Popping a stack frame!")


   if __name__ == "__main__":
       with SysTracer:
           def f():
               def g():
                   return 42
               return g()
           # push, push, pop, pop
           answer_to_life_universe_everything = f()

No ``pyc.exec`` required
------------------------

Notice that we did **not** need :func:`pyccolo.exec` here, because Python's
built-in tracing does not involve any AST-level transformation — the handlers
attach through ``sys.settrace`` at runtime, not by rewriting source. (If we had
*also* registered handlers for AST events such as ``pyc.before_stmt``, we would
need ``pyc.exec`` to instrument code in the same file where the tracer is
defined; see :doc:`quickstart`.)

.. note::

   The ``opcode`` event requires **Python >= 3.7**.

Composing with an existing ``settrace`` function
------------------------------------------------

Pyccolo composes with an existing ``sys.settrace(...)`` function, too. Its unit
tests for ``call`` and ``return_`` pass even when
`coverage.py <https://coverage.readthedocs.io/>`_ is active — and without
breaking it — even though coverage.py also drives Python's built-in tracing.
