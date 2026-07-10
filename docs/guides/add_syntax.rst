Add new surface syntax
======================

Pyccolo can teach Python syntax the parser would normally reject — ``?.`` optional
chaining, a ``|>`` pipe operator, ``name{ ... }`` blocks — and give it runtime
meaning. The trick is a two-step move: a source-level rewrite turns your new token
into something the parser accepts, and Pyccolo *remembers where the rewrite
happened* so a handler can recognize the resulting node and act on it.

.. note::

   Syntax augmentation requires Python ≥ 3.8.

Declare a rewrite with ``AugmentationSpec``
-------------------------------------------

A tracer declares new syntax with an :class:`~pyccolo.AugmentationSpec` class
attribute: a token → replacement rewrite, tagged with *where* the token sits so
Pyccolo can re-associate it after the parser has run. Optional chaining, for
instance, rewrites ``?.`` down to a plain ``.``:

.. code-block:: python

   import pyccolo as pyc


   class OptionalChainer(pyc.BaseTracer):
       optional_chaining_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
       )

Inside a handler, ask which specs produced a given node with
:meth:`~pyccolo.BaseTracer.get_augmentations`, keyed on the node's id — that is how
you tell an ordinary ``.`` apart from one that came from ``?.``.

A complete example: a pipe operator
------------------------------------

Here is the whole loop end to end. We declare ``|>`` as a **binop**-position token
(rewritten to Python's real ``|``), then handle ``before_binop`` for nodes our
spec produced, returning a two-argument function that applies the right operand to
the left:

.. testcode::

   import pyccolo as pyc


   class Pipe(pyc.BaseTracer):
       global_guards_enabled = False        # pure syntax; no runtime guards needed

       pipe_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
       )

       @pyc.before_binop(when=lambda node: isinstance(node.op, ast.BitOr))
       def apply(self, ret, node, *_, **__):
           if self.pipe_spec in self.get_augmentations(id(node)):
               return lambda left, right: right(left)   # x |> f  ==  f(x)
           return ret


   with Pipe:
       assert pyc.exec("result = (1, 2, 3) |> list")["result"] == [1, 2, 3]

Returning a function from ``before_binop`` **replaces the operation itself** — the
``|`` we rewrote to is never actually a bitwise-or. The ``when=`` guard keeps us
off ordinary ``|`` expressions, and ``get_augmentations`` narrows us to just the
``|>`` occurrences. The :doc:`/tutorials/pipeline_operator` builds this up from
scratch and adds an assigning ``|>>`` variant.

The augmentation vocabulary
---------------------------

``AugmentationType`` names where a token sits, so Pyccolo knows how to rewrite it
to something legal and re-associate it afterward:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - ``AugmentationType``
     - Position, with an example token
   * - ``prefix`` / ``suffix``
     - a token before / after an expression
   * - ``dot_prefix`` / ``dot_suffix``
     - around an attribute dot — e.g. ``?.`` (``dot_suffix``)
   * - ``binop``
     - a binary-operator position — e.g. ``|>`` rewritten to ``|``
   * - ``boolop``
     - a boolean-operator position — e.g. ``??`` rewritten to ``or``
   * - ``call`` / ``subscript``
     - a call ``(...)`` or subscript ``[...]`` position
   * - ``custom``
     - a fully custom rewrite (see below)

A ``boolop`` rewrite is a plain token swap at the source level:

.. testcode::

   class Coalesce(pyc.BaseTracer):
       global_guards_enabled = False
       nullish_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.boolop, token="??", replacement=" or "
       )

       @pyc.before_boolop_arg
       def keep(self, thunk, *_, **__):
           return thunk


   with Coalesce:
       assert pyc.transform("y = a ?? b") == "y = a  or  b"

(giving ``??`` true nullish-coalescing behavior — return the left unless it is
``None`` — means handling ``before_boolop_arg`` with a thunk, as the full
``optional_chaining.py`` example does.)

Paired delimiters: brace blocks
-------------------------------

Beyond single-token replacement, a spec can describe a **paired delimiter** by
setting ``close_token`` / ``close_replacement``. This is the basis for
statement-bodied ``name{ ... }`` blocks — ``map{ ... }``, ``do{ ... }``,
``run{ ... }``. The helper
:func:`~pyccolo.syntax_augmentation.make_paired_delimiter_augmenter` wires up the
common case:

.. code-block:: python

   from pyccolo.syntax_augmentation import make_paired_delimiter_augmenter

   # turn `name{ body }` into a call that receives the block body
   augmenter = make_paired_delimiter_augmenter(
       triggers=["map", "do"], emit=my_emit, open_tok="{", close_tok="}"
   )

The shipped ``block_lambda.py``, ``func_block.py``, and ``brace_subscript.py``
examples are three different takes on brace blocks — value-snapshot closures, a
function-wrapping ``body_func_wrapper``, and a pure ``{`` ↔ ``[`` swap
respectively.

Context-sensitive rewrites
--------------------------

When a single token → token substitution can't express the rewrite — when whether
to rewrite depends on the surrounding source — subclass
:class:`~pyccolo.CustomRewrite` and reference it from a spec with
``aug_type=pyc.AugmentationType.custom``.

The full augmentation API is in :doc:`/reference/augmentation`, and the
`pipescript <https://github.com/smacke/pipescript>`_ project is the largest
real-world showcase, layering a whole pipe-and-placeholder dialect on top of
Python.
