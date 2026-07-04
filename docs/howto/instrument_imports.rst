Instrument imported modules
===========================

**Goal:** apply instrumentation to code in *other* modules, at import time.

Instrumentation is opt-in for modules imported within a tracing context. To choose
which modules get instrumented, override
:meth:`~pyccolo.BaseTracer.should_instrument_file`, which is called with each
module's filename:

.. code-block:: python

   import pyccolo as pyc

   class MyTracer(pyc.BaseTracer):
       def should_instrument_file(self, filename: str) -> bool:
           return filename.endswith("foo.py")

       @pyc.register_handler(pyc.after_assign_rhs)
       def handle(self, ret, *_, **__):
           return ret + 1

   with MyTracer:
       import foo   # contents of `foo` get instrumented
       import bar   # contents of `bar` do not

To instrument *everything* imported inside the context, set the class attribute
``instrument_all_files = True`` instead of overriding the method (see
:doc:`/reference/config`).

How it works
------------

Imports are instrumented by registering a custom finder / loader on
``sys.meta_path``. The loader **ignores** cached bytecode (which may be
uninstrumented) and avoids writing **new** cached bytecode (which would be
instrumented, and could cause confusion later when instrumentation is not
desired).

Running a whole workload under a tracer
---------------------------------------

To drive an import-heavy workload (for example, a test suite) under a tracer,
install the import hook explicitly:

.. code-block:: python

   from pyccolo.import_hooks import patch_meta_path

   tracer = MyTracer.instance()
   with tracer:
       with patch_meta_path(pyc._TRACER_STACK):
           ...  # e.g. pytest.console_main()

This is exactly how the bundled ``coverage.py`` example runs pytest under
instrumentation.

.. seealso::

   :doc:`/reference/cli` — the ``pyc`` tool runs a script through this same import
   machinery, which is why script tracers must opt their own file in.
