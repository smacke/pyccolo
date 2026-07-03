Performance and guards
======================

Pyccolo instrumentation adds significant overhead to Python. In many cases the
overhead can be partially mitigated: if, for example, you only need
instrumentation the *first* time a statement runs, you can deactivate it
afterwards so that further calls (or loop iterations) execute uninstrumented,
with all the mighty performance of native Python.

This is implemented with **guards** — runtime flags associated with a function
or loop that gate whether its instrumentation fires. Activating a guard turns the
instrumented region off; deactivating it turns it back on.

.. code-block:: python

   import pyccolo as pyc


   class TracesOnce(pyc.BaseTracer):
       @pyc.register_raw_handler((pyc.after_for_loop_iter, pyc.after_while_loop_iter))
       def after_loop_iter(self, *_, guard, **__):
           self.activate_guard(guard)

       @pyc.register_raw_handler(pyc.after_function_execution)
       def after_function_exec(self, *_, guard, **__):
           self.activate_guard(guard)

Here, after the first loop iteration (respectively, the first time a function
finishes executing), the associated guard is activated, so subsequent iterations
and calls run without instrumentation. Note the use of
:func:`pyccolo.register_raw_handler`: guard-bearing events pass the guard name as
the ``guard`` keyword argument, and the raw registrar is the form that surfaces
it.

Subsequent calls / iterations will be instrumented again only after calling
:meth:`self.deactivate_guard(...) <pyccolo.BaseTracer.deactivate_guard>` on the
associated function / loop guard.

Guards are one of the levers that make heavyweight dynamic analyses practical on
real workloads: pay for instrumentation exactly where and when you need the
signal, and run native everywhere else.
