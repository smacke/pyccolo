Instrumenting imported modules
==============================

By default, only code you explicitly hand to :func:`pyccolo.exec` /
:func:`pyccolo.eval` is instrumented. Instrumentation is *opt-in* for modules
imported within a tracing context, and a tracer decides which files to
instrument by overriding :meth:`~pyccolo.BaseTracer.should_instrument_file`:

.. code-block:: python

   import pyccolo as pyc


   class MyTracer(pyc.BaseTracer):
       def should_instrument_file(self, filename: str) -> bool:
           return filename.endswith("foo.py")

       # handlers, etc. defined below
       ...


   with MyTracer:
       import foo  # contents of `foo` get instrumented
       import bar  # contents of `bar` do not

The method is called with the candidate module's filename and returns whether it
should be rewritten. (To instrument *everything* imported inside the context,
set the class attribute ``instrument_all_files = True`` instead of overriding the
method.)

How it works
------------

Imports are instrumented by registering a custom finder / loader on
:data:`sys.meta_path`. This loader:

- **ignores cached bytecode**, which may be uninstrumented (an already-compiled
  ``.pyc`` would otherwise be used verbatim, skipping the rewrite); and
- **avoids generating new cached bytecode**, which would be *instrumented* —
  writing it to disk could cause confusion later, when instrumentation is not
  desired.

The finder/loader is installed and removed with the tracing context, so ordinary
imports outside the ``with`` block behave exactly as they always do.

A complete example — lazy imports that resolve on first use — ships as
`pyccolo/examples/lazy_imports.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/lazy_imports.py>`_
(see the :doc:`examples`).
