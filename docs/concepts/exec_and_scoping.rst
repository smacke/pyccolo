Why ``pyc.exec``, and how scoping works
=======================================

Newcomers almost always ask the same question: *why do I have to wrap my code in*
``pyc.exec("...")`` *instead of just writing it normally?* The answer is a
consequence of when Python compiles code, and it is worth understanding because it
also tells you exactly when you **don't** need ``pyc.exec``.

The AST is fixed at compile time
--------------------------------

A module's abstract syntax tree is built once, when the module is compiled and
imported — and Pyccolo's instrumentation is an AST rewrite (see
:doc:`/concepts/model`). When your script first starts running, the
module containing your tracer is *already compiled*, and the tracer was not active
at that moment. So any ordinary, unquoted Python written in the same file as the
tracer definition has no instrumentation woven into it — there was no active tracer
to weave it in.

To instrument code that lives in that file, you "quote" it as a string (or an AST)
and hand it to ``pyc.exec`` or ``pyc.eval``. These parse it, run it through the
instrumentation pipeline, and execute the result — this time with your tracer
active:

.. code-block:: python

   import pyccolo as pyc


   class HelloTracer(pyc.BaseTracer):
       @pyc.before_stmt
       def handle_stmt(self, *_, **__):
           print("Hello, world!")


   with HelloTracer:
       pyc.exec("for _ in range(10): pass")  # instrumented; prints 11 times

``pyc.exec`` returns a namespace
--------------------------------

``pyc.exec`` runs its code in a sandbox and returns the resulting namespace as a
dict, so you can inspect what the executed code produced:

.. testcode::

   env = pyc.exec("x = 42")
   assert env["x"] == 42

``pyc.eval`` is the expression-valued counterpart, returning the value of a single
expression. Both compile under a synthetic sandbox filename so their frames are
recognizable; if you need those frames visible in tracebacks, see the
traceback-visibility helpers in :doc:`/reference/utilities`.

When you *don't* need ``pyc.exec``
----------------------------------

Two important cases run instrumented code without any quoting:

- **``sys`` events.** Handlers for ``call``, ``line``, ``return_``, ``exception``,
  and ``opcode`` ride on Python's built-in tracing machinery, not on an AST
  rewrite. Nothing needs to be recompiled, so ordinary in-file code triggers them
  directly. See :doc:`/guides/tracing_real_programs`.

- **Imported modules.** Code imported *inside* a tracing context can be
  instrumented at import time, because Pyccolo controls the import and can rewrite
  the module as it loads — provided the module opts in via
  :meth:`~pyccolo.BaseTracer.should_instrument_file`. See
  :doc:`/guides/tracing_real_programs`.

Reaching into the caller's scope
--------------------------------

Handlers receive the executing ``frame``, and sometimes need to write a value back
into it — for example to set up a guard variable or bind a synthetic name.
``pyc.set_frame_local`` does this safely across the frame's local and global
namespaces, rather than mutating ``frame.f_locals`` directly (which does not
reliably propagate). It is part of the :doc:`utilities reference </reference/utilities>`,
alongside the sandbox-filename constants used to identify ``pyc.exec`` frames. For
the full set of run-and-compile entry points, see :doc:`/reference/tracer`.
