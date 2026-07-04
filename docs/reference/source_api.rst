Source-to-source API
====================

.. currentmodule:: pyccolo

When you want the rewritten *source* — for a linter, formatter, or source map —
rather than a running program, use these module-level functions. They operate
against the active tracer stack (or the tracers you pass explicitly). See
:doc:`/howto/source_to_source` for the how-to.

.. autofunction:: pyccolo.transform

.. autofunction:: pyccolo.untransform

.. autofunction:: pyccolo.parse

Both ``transform`` and ``untransform`` accept ``positions=[(line, col), ...]`` and
return the remapped positions alongside the rewritten source, for source-map-style
tooling.

Pure (analysis-only) transforms
-------------------------------

Passing ``pure=True`` marks a transform whose result will never be executed —
purely for analysis. Cooperating rewrites can consult
:func:`is_pure_transform` to avoid touching execution-relevant state. It is
thread- and async-safe (backed by a :class:`~contextvars.ContextVar`).

.. autofunction:: pyccolo.is_pure_transform

.. code-block:: python

   with OptionalChainer:
       assert pyc.transform("y = a?.b?.c") == "y = a.b.c"
       tree = pyc.parse("y = a?.b?.c", instrument=False)
       assert pyc.untransform(tree) == "y = a?.b?.c"
       assert pyc.transform("y = a?.b", pure=True) == "y = a.b"

Sandbox filenames
-----------------

Instrumented code run through :meth:`~pyccolo.BaseTracer.exec` /
:meth:`~pyccolo.BaseTracer.eval` is compiled under a synthetic filename so it can
be recognized in frames and tracebacks:

.. data:: pyccolo.SANDBOX_FNAME

   The filename assigned to sandboxed ``exec``/``eval`` code.

.. data:: pyccolo.SANDBOX_FNAME_PREFIX

   The prefix shared by all sandbox filenames — test ``frame.f_code.co_filename``
   against it to detect sandboxed frames.

.. data:: pyccolo.TRACEBACK_VISIBLE_SANDBOX_FILES

   The registry of sandbox files whose frames should remain visible in
   tracebacks (see :func:`~pyccolo.mark_traceback_visible`).
