Add new surface syntax
======================

**Goal:** teach Python a new bit of syntax and give it runtime meaning.

.. note::

   Syntax augmentation requires Python ≥ 3.8.

A tracer declares new syntax with an ``AugmentationSpec`` class attribute. A spec
is a source-level token → replacement rewrite; Pyccolo remembers *where* the
rewrite happened, so a handler can attach to the resulting AST node. For example,
JavaScript-style optional chaining rewrites ``?.`` down to a plain ``.``:

.. code-block:: python

   import pyccolo as pyc

   class OptionalChainer(pyc.BaseTracer):
       optional_chaining_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
       )

Inside a handler, ask which specs produced a node with
:meth:`~pyccolo.BaseTracer.get_augmentations`:

.. code-block:: python

   @pyc.register_handler(pyc.before_attribute_load)
   def handle(self, obj, node, *_, **__):
       if self.optional_chaining_spec in self.get_augmentations(id(node)):
           ...  # this `.` came from a `?.`

Kinds of augmentation
---------------------

``AugmentationType`` names where a token sits, so Pyccolo can rewrite it to
something legal and re-associate it afterward:

- ``prefix`` / ``suffix`` — a token before / after an expression;
- ``dot_prefix`` / ``dot_suffix`` — around an attribute dot (e.g. ``?.``);
- ``binop`` — a binary-operator position (e.g. ``|>``, rewritten to ``|``);
- ``boolop`` — a boolean-operator position (e.g. ``??``, rewritten to ``or``);
- ``call`` / ``subscript`` — call / subscript positions;
- ``custom`` — a fully custom rewrite (see below).

Paired delimiters (brace blocks)
--------------------------------

Beyond single-token replacement, a spec can describe a **paired delimiter** by
setting ``close_token`` / ``close_replacement`` — the basis for statement-bodied
``name{ ... }`` blocks. See the ``block_lambda``, ``func_block``, and
``brace_subscript`` examples.

Context-sensitive rewrites
--------------------------

When a single token→token substitution can't express the rewrite, subclass
``CustomRewrite`` and reference it from a spec with
``aug_type=pyc.AugmentationType.custom``.

.. seealso::

   :doc:`/tutorials/optional_chaining` walks through a full build; the
   `pipescript <https://github.com/smacke/pipescript>`_ project is the largest
   showcase, layering a pipe-and-placeholder dialect on top of Python. Full API in
   :doc:`/reference/augmentation`.
