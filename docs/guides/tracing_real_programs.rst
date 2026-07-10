Tracing real programs
=====================

The quickstart runs code through :func:`pyccolo.exec` because a tracer can't
instrument source that was already compiled before it became active — and in a
script, the tracer's *own* module is compiled first. But most real programs are
not a string you hand to ``pyc.exec``: they are imported modules, ordinary
functions, or a whole application launched from the command line. This guide
covers the four ways to get a tracer onto real code: **imports**, the
**instrumented decorator**, **sys.settrace-style events**, and the **CLI**.

Instrument imported modules
---------------------------

Modules imported *inside* a tracing context can be instrumented at import time.
Because you rarely want to rewrite every dependency, it is opt-in: override
:meth:`~pyccolo.BaseTracer.should_instrument_file`, called with each module's
filename, and return ``True`` for the files you want:

.. code-block:: python

   import pyccolo as pyc


   class MyTracer(pyc.BaseTracer):
       def should_instrument_file(self, filename: str) -> bool:
           return filename.endswith("mypackage/core.py")

       @pyc.after_assign_rhs
       def handle(self, ret, *_, **__):
           return ret + 1


   with MyTracer:
       import mypackage.core   # its assignments are instrumented
       import numpy            # numpy is left untouched

To instrument *everything* imported inside the context, set the class attribute
``instrument_all_files = True`` instead of overriding the method.

Under the hood, Pyccolo installs a finder/loader on :data:`sys.meta_path`. The
loader deliberately **ignores** cached ``.pyc`` bytecode (which might be
uninstrumented) and avoids **writing** new cache files (which would be
instrumented, and could confuse a later run that doesn't want tracing).

To drive an import-heavy workload — a test suite, say — under a tracer, install
the import hook explicitly with ``patch_meta_path``:

.. code-block:: python

   from pyccolo.import_hooks import patch_meta_path

   with MyTracer.instance():
       with patch_meta_path(pyc._TRACER_STACK):
           pytest.console_main()

This is exactly how the bundled ``coverage.py`` example measures coverage of a
pytest run; the :doc:`/tutorials/coverage_tracer` builds it up step by step.

Instrument a single function
-----------------------------

When you just want one function traced — no imports, no ``pyc.exec`` — decorate it
with :func:`pyccolo.instrumented`, passing the tracer instances to apply. Pyccolo
recompiles the function's own source with instrumentation and runs it under those
tracers:

.. testcode::

   import pyccolo as pyc


   class AddOne(pyc.BaseTracer):
       @pyc.after_assign_rhs
       def handle(self, ret, *_, **__):
           return ret + 1


   @pyc.instrumented([AddOne.instance()])
   def compute():
       x = 10
       return x


   assert compute() == 11

By default the decorator leaves the original function object alone and returns an
instrumented copy; pass ``mutate=True`` to rewrite the original in place. Multiple
tracers compose here just as they do with nested ``with`` blocks —
``@pyc.instrumented([A.instance(), B.instance()])``.

sys.settrace-style events
-------------------------

Not every event comes from AST rewriting. Pyccolo also surfaces Python's built-in
`sys.settrace <https://docs.python.org/3/library/sys.html#sys.settrace>`_ hooks —
``call``, ``line``, ``return_``, ``exception``, and ``opcode`` — as ordinary
events. These fire on *any* code running in the traced context, so you do **not**
need ``pyc.exec`` or import instrumentation to observe them:

.. testcode::

   pushes_and_pops = []


   class FrameTracer(pyc.BaseTracer):
       @pyc.register_handler(pyc.call)
       def on_call(self, _ret, _node, frame, *_, **__):
           pushes_and_pops.append(("call", frame.f_code.co_name))

       @pyc.register_handler(pyc.return_)
       def on_return(self, _ret, _node, frame, *_, **__):
           pushes_and_pops.append(("return", frame.f_code.co_name))


   def g():
       return 42


   with FrameTracer:
       g()

   assert ("call", "g") in pushes_and_pops
   assert ("return", "g") in pushes_and_pops

Because these ride on the interpreter's own tracing, they compose with an
*existing* ``sys.settrace`` function: Pyccolo's ``call``/``return_`` events keep
working even when `coverage.py <https://coverage.readthedocs.io/>`_ is active, and
without breaking it. (``opcode`` events require Python ≥ 3.7.) In the
:doc:`event catalog </reference/events>` these carry the **sys** flag.

Run a whole script under the CLI
--------------------------------

Pyccolo ships a ``pyc`` command that runs a script or module through the same
import machinery, under one or more tracers named by dotted path:

.. code-block:: bash

   # run a script
   pyc myscript.py -t mypackage.tracers.MyTracer

   # run a module (like python -m)
   pyc -m mypackage -t mypackage.tracers.MyTracer

   # compose several tracers; they layer left to right
   pyc myscript.py -t pkg.TracerA -t pkg.TracerB

Because the CLI loads your script *through* the import hook, a tracer used this
way must opt its own file in — return ``True`` from ``should_instrument_file`` (or
set ``instrument_all_files = True``). That is why the shipped examples come in
``Script*`` variants, e.g. ``ScriptOptionalChainer``. Full CLI reference:
:doc:`/reference/cli`.
