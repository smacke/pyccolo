Syntax augmentation
===================

.. currentmodule:: pyccolo

These are the building blocks for defining *new surface syntax* — see
:doc:`/howto/add_syntax` for the how-to and :doc:`/tutorials/optional_chaining`
for a full worked example. Syntax augmentation requires Python ≥ 3.8.

``AugmentationSpec``
--------------------

.. autoclass:: pyccolo.AugmentationSpec
   :members:

A spec declares a source-level rewrite from an (otherwise illegal) token to a
legal one; Pyccolo records *where* the rewrite happened so a handler can attach to
the resulting node via :meth:`~pyccolo.BaseTracer.get_augmentations`. For
paired-delimiter (brace-block) syntax, also set ``close_token`` /
``close_replacement``; ``is_paired`` and ``is_custom`` report the kind.

.. code-block:: python

   optional_chaining_spec = pyc.AugmentationSpec(
       aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
   )

``AugmentationType``
--------------------

.. autoclass:: pyccolo.AugmentationType
   :members:
   :undoc-members:

The kind of token position a spec rewrites — e.g. ``dot_suffix`` (``?.``),
``binop`` (``|>``), ``boolop`` (``??``), or ``custom``.

``CustomRewrite``
-----------------

.. autoclass:: pyccolo.CustomRewrite
   :members:

The extension point for context-sensitive rewrites that a single token→token
substitution can't express. Subclass it and reference it from a spec with
``aug_type=AugmentationType.custom``.

Positions
---------

.. autoclass:: pyccolo.Position
   :members:

``pyc.transform`` / ``pyc.untransform`` accept and return ``Position``\\ s so
tooling can build source maps across the rewrite.

The rewriter
------------

.. autoclass:: pyccolo.AstRewriter
   :members:
