Source-to-source transforms
===========================

Everything so far has instrumented code in order to *run* it. But the same
machinery can hand you the rewritten **source** without executing anything —
which is what a linter, a formatter, a source-map generator, or a compile-to-plain
Python tool needs. :func:`pyccolo.transform` desugars, and
:func:`pyccolo.untransform` resugars.

Desugar and resugar
-------------------

Under a tracer that defines syntax augmentation, ``pyc.transform`` returns the
instrumented / desugared source as a string, and ``pyc.untransform`` reverses an
augmentation — turning valid Python back into the augmented surface syntax. Using
the shipped optional-chaining example, ``a?.b?.c`` desugars to plain ``a.b.c``
and back:

.. testcode::

   import pyccolo as pyc
   from pyccolo.examples.optional_chaining import ScriptOptionalChainer as OC

   with OC:
       assert pyc.transform("y = a?.b?.c") == "y = a.b.c"

       tree = pyc.parse("y = a?.b?.c", instrument=False)
       assert pyc.untransform(tree) == "y = a?.b?.c"

``pyc.parse(code, instrument=False)`` gives you the augmented syntax parsed into
an AST *without* adding instrumentation, which is the right input to
``untransform``.

Analysis-only (pure) transforms
-------------------------------

Some rewrites keep bookkeeping state as a side effect of running — fine when you
are about to execute the result, wrong when you only want to inspect it. Pass
``pure=True`` to mark a transform as analysis-only, so cooperating rewrites know
not to disturb execution-relevant state:

.. testcode::

   with OC:
       assert pyc.transform("y = a?.b", pure=True) == "y = a.b"

A rewrite can consult :func:`pyccolo.is_pure_transform` to branch on this. It is
thread- and async-safe, backed by a :class:`~contextvars.ContextVar`, so a pure
transform on one thread won't perturb a live tracer on another.

Source maps
-----------

For tooling that must map positions between the original and transformed source —
underlining an error at the right column, say — both ``transform`` and
``untransform`` accept a ``positions=[(line, col), ...]`` argument and return the
corresponding positions in the transformed (or untransformed) coordinate space
alongside the source:

.. code-block:: python

   with OC:
       transformed, mapped = pyc.transform(
           "y = a?.b?.c", positions=[(1, 4)]
       )
       # `mapped` holds the (line, col) of that point in `transformed`

The precise signatures — including what ``mapped`` contains — are in
:doc:`/reference/source_api`.
