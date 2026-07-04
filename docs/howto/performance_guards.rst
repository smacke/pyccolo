Reduce instrumentation overhead with guards
===========================================

**Goal:** stop paying for instrumentation once you no longer need it.

Instrumentation adds significant overhead. Often you only need it the *first* time
a statement runs — the first iteration of a loop, or the first call of a function.
**Guards** let you switch instrumentation off after that point, so later
iterations / calls run native, uninstrumented code.

Each guarded event passes a ``guard`` keyword argument; call
:meth:`~pyccolo.BaseTracer.activate_guard` with it to deactivate the enclosing
region:

.. code-block:: python

   import pyccolo as pyc

   class TracesOnce(pyc.BaseTracer):
       @pyc.register_raw_handler((pyc.after_for_loop_iter, pyc.after_while_loop_iter))
       def after_loop_iter(self, *_, guard, **__):
           self.activate_guard(guard)

       @pyc.register_raw_handler(pyc.after_function_execution)
       def after_function_exec(self, *_, guard, **__):
           self.activate_guard(guard)

After the first iteration / execution, the associated loop or function runs
uninstrumented. To turn instrumentation back on for that region, call
:meth:`~pyccolo.BaseTracer.deactivate_guard` with the same guard.

Transformational tracers
------------------------

Tracers that only *rewrite* syntax (rather than emit runtime events) set the class
attribute ``global_guards_enabled = False`` so they layer cleanly with others; see
:doc:`/reference/config`.

.. seealso::

   :doc:`/concepts/rewrite_pipeline` explains where guards sit in the emit path,
   and :doc:`/reference/tracer` documents the guard methods.
