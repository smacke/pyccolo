Syntax augmentation
===================

Pyccolo can go beyond instrumenting existing Python: a tracer can define *new
surface syntax*. It does this with an :class:`pyccolo.AugmentationSpec`, which
declares a source-level token → replacement rewrite. Pyccolo remembers *where*
the rewrite happened, so a handler can attach to the resulting AST node.

.. note::

   Syntax augmentation requires **Python >= 3.8**.

A first example: optional chaining
----------------------------------

JavaScript-style optional chaining rewrites ``?.`` down to a plain ``.``, then
resolves the access to ``None`` whenever the receiver is ``None``. The
augmentation spec that drives it is a single declaration:

.. code-block:: python

   import pyccolo as pyc

   optional_chaining_spec = pyc.AugmentationSpec(
       aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
   )

A complete, tested implementation of optional chaining and nullish coalescing
(``??``) ships in
`pyccolo/examples/optional_chaining.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/optional_chaining.py>`_:

.. code-block:: python

   import pyccolo as pyc
   from pyccolo.examples.optional_chaining import ScriptOptionalChainer

   with ScriptOptionalChainer:
       pyc.exec("bar = None\nprint(bar?.foo)")  # -> None

Augmentation types
------------------

:class:`pyccolo.AugmentationType` enumerates where a token may be anchored:

- ``prefix`` / ``suffix`` — before / after an identifier or expression;
- ``dot_prefix`` / ``dot_suffix`` — around an attribute-access dot (as in the
  ``?.`` example);
- ``binop`` — a binary-operator position (as with the ``|>`` pipeline operator);
- ``custom`` — a context-sensitive rewrite expressed via
  :class:`pyccolo.CustomRewrite` (below).

The mechanism works by rewriting an *illegal* token span (e.g. ``?.`` or ``|>``)
into a *legal* one (``.`` or bitwise-or ``|``), parsing the now-valid Python,
and then associating the resulting AST node with the augmentation so the
corresponding handler runs. Because a single event-emission transform is shared
by every handler that cares about it, features compose without conflicting AST
rewrites (see :doc:`composing_tracers`).

A larger showcase: pipescript
-----------------------------

The most complete demonstration of syntax augmentation is
`pipescript <https://github.com/smacke/pipescript>`_, which layers a whole
pipe-and-placeholder dialect on top of Python:

.. code-block:: text

   # in IPython / Jupyter, after `%load_ext pipescript`
   result = arrays |> map[$
     |> $array[np.isfinite($array)]
     |> np.abs
     |> np.max($, initial=1.0)
   ] |> max

Under the hood, pipescript rewrites illegal token spans like ``|>`` to legal ones
(here, bitwise-or ``|``), then uses Pyccolo to associate the resulting
:class:`ast.BinOp` with the ``|>`` operator and run the corresponding handler.

Beyond single-token replacement
-------------------------------

An :class:`~pyccolo.AugmentationSpec` also supports **paired-delimiter**
(brace-block) augmentation via its ``close_token`` / ``close_replacement``
fields (its ``is_paired`` property reports whether they are set). This powers
statement-bodied ``name{ ... }`` blocks — see the ``block_lambda``,
``func_block``, and ``brace_subscript`` entries in the :doc:`examples`.

For rewrites the token and paired passes cannot express, the ``custom`` type and
:class:`pyccolo.CustomRewrite` provide a context-sensitive extension point.

Full coverage of the augmentation surface (including transform/untransform round
trips and position remapping) lives in
`test/test_syntax_augmentation.py <https://github.com/smacke/pyccolo/blob/master/test/test_syntax_augmentation.py>`_.
See also :doc:`source_to_source` for turning augmented syntax into plain Python
source and back.
