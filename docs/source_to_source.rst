Source-to-source: transform, untransform, and pure mode
=======================================================

Sometimes you want the rewritten *source*, not a running program — for a linter,
a formatter, or a source map. Pyccolo exposes its rewriting machinery directly:

- :func:`pyccolo.transform` returns instrumented / desugared source;
- :func:`pyccolo.untransform` reverses an augmentation, *resugaring* valid Python
  back into the augmented syntax;
- :func:`pyccolo.parse` returns the (optionally instrumented) AST.

.. code-block:: python

   import pyccolo as pyc
   from pyccolo.examples.optional_chaining import ScriptOptionalChainer as OC

   with OC:
       # desugar augmented syntax down to plain, valid Python:
       assert pyc.transform("y = a?.b?.c") == "y = a.b.c"

       # ...and resugar it back from the parsed tree:
       tree = pyc.parse("y = a?.b?.c", instrument=False)
       assert pyc.untransform(tree) == "y = a?.b?.c"

       # pure=True marks an analysis-only transform (no runtime side effects):
       assert pyc.transform("y = a?.b", pure=True) == "y = a.b"

Position remapping
------------------

Both :func:`~pyccolo.transform` and :func:`~pyccolo.untransform` accept a
``positions=[(line, col), ...]`` argument and return the remapped positions in
the transformed (or untransformed) coordinate space — everything you need for
source-map-style tooling that has to point back at the original text.

Pure mode
---------

Passing ``pure=True`` signals an **analysis-only** transform whose result is
never executed. Cooperating rewrites can consult :func:`pyccolo.is_pure_transform`
to avoid touching execution-relevant state (for example, skipping the injection
of runtime event-emission calls when only the desugared shape is wanted). Pure
mode is tracked with a :class:`~contextvars.ContextVar`, so it is safe to use
across threads and ``async`` tasks.

This is the same rewriting pipeline that :doc:`syntax_augmentation` builds on;
``transform`` / ``untransform`` just expose it without executing the result.
