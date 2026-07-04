Get transformed source instead of running it
============================================

**Goal:** obtain the rewritten *source*, not a running program — for a linter,
formatter, or source map.

``pyc.transform(code)`` returns instrumented / desugared source, and
``pyc.untransform(tree)`` reverses an augmentation, resugaring valid Python back
into the augmented syntax:

.. code-block:: python

   import pyccolo as pyc
   from pyccolo.examples.optional_chaining import ScriptOptionalChainer as OC

   with OC:
       # desugar augmented syntax down to plain, valid Python:
       assert pyc.transform("y = a?.b?.c") == "y = a.b.c"

       # ...and resugar it back from the parsed tree:
       tree = pyc.parse("y = a?.b?.c", instrument=False)
       assert pyc.untransform(tree) == "y = a?.b?.c"

Source maps
-----------

Both ``transform`` and ``untransform`` accept ``positions=[(line, col), ...]`` and
return the remapped positions in the transformed (or untransformed) coordinates,
for source-map-style tooling.

Analysis-only (pure) transforms
-------------------------------

Passing ``pure=True`` marks an analysis-only transform whose result is never
executed:

.. code-block:: python

   with OC:
       assert pyc.transform("y = a?.b", pure=True) == "y = a.b"

Cooperating rewrites can consult ``pyc.is_pure_transform()`` to avoid touching
execution-relevant state during such a transform. It is thread- and async-safe,
backed by a :class:`~contextvars.ContextVar`.

.. seealso::

   :doc:`/reference/source_api` for the full signatures.
