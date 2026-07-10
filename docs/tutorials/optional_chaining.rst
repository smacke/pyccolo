Tutorial: build optional chaining (``?.``)
==========================================

In this tutorial you'll teach Python a brand-new piece of syntax: JavaScript-style
**optional chaining**, where ``foo?.bar`` short-circuits to ``None`` whenever
``foo`` is ``None`` instead of raising :class:`AttributeError`. You'll see how
Pyccolo lets you introduce surface syntax that the Python parser would normally
reject, and attach behavior to it.

This builds the essence of the ``optional_chaining.py`` example. Syntax
augmentation requires Python ≥ 3.8.

Step 1: declare the new syntax
------------------------------

``?.`` is not legal Python, so first we tell Pyccolo how to rewrite it into
something that *is*. An ``AugmentationSpec`` declares a token-level rewrite — here,
turn ``?.`` into a plain ``.`` — and Pyccolo remembers **where** each rewrite
happened so a handler can later recognize the affected node. Declare it as a class
attribute:

.. code-block:: python

   import ast

   import pyccolo as pyc


   class OptionalChainer(pyc.BaseTracer):
       optional_chaining_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
       )

``aug_type=dot_suffix`` says the token sits where a ``.`` attribute-access dot
would go. After this rewrite, ``bar?.foo`` parses as an ordinary
``ast.Attribute`` load — but one that Pyccolo has tagged as originating from
``?.``. See :doc:`/guides/add_syntax` and :doc:`/reference/augmentation` for the
full augmentation vocabulary.

Step 2: handle the tagged access
--------------------------------

Now attach behavior. When an attribute is about to be dereferenced, Pyccolo fires
``before_attribute_load`` with the receiver object. We check whether *this*
attribute node came from our ``?.`` spec using
:meth:`~pyccolo.BaseTracer.get_augmentations`, and if the receiver is ``None``, we
substitute a sentinel that makes any further attribute access also yield ``None``:

.. code-block:: python

   class OptionalChainer(pyc.BaseTracer):
       class ResolvesToNone:
           def __getattr__(self, _item):
               return self  # keep resolving to the sentinel down the chain

       resolves_to_none = ResolvesToNone()

       optional_chaining_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
       )

       @pyc.register_handler(pyc.before_attribute_load)
       def handle_before_attr(self, obj, node: ast.Attribute, *_, **__):
           if (
               self.optional_chaining_spec in self.get_augmentations(id(node))
               and obj is None
           ):
               return self.resolves_to_none
           return obj

Returning a value from ``before_attribute_load`` **replaces** the receiver that
gets dereferenced (see :doc:`/reference/handlers`). The sentinel's ``__getattr__``
returns itself, so an entire chain like ``a?.b?.c`` keeps resolving through the
sentinel rather than blowing up.

Step 3: turn the sentinel back into ``None``
--------------------------------------------

The sentinel is an implementation detail; the *result* the user sees should be a
real ``None``. A whole attribute/subscript/call chain is wrapped by a compound
"load complex symbol" event, so we intercept its result and, if it's our
sentinel, override it with ``pyc.Null`` — the way a handler says "replace this
value with an actual ``None``" (returning ``None`` would mean "don't override"):

.. code-block:: python

       @pyc.register_raw_handler(pyc.after_load_complex_symbol)
       def handle_after_load_complex_symbol(self, ret, *_, **__):
           if isinstance(ret, self.ResolvesToNone):
               return pyc.Null
           return ret

Step 4: run it
--------------

Because the tracer defines AST-level syntax, run the target code through
``pyc.exec`` so it gets instrumented:

.. code-block:: python

   with OptionalChainer:
       pyc.exec("bar = None\nprint(bar?.foo)")  # -> None

To instrument files loaded through the import machinery or the ``pyc``
command-line tool instead, subclass and opt those files in by overriding
:meth:`~pyccolo.BaseTracer.should_instrument_file`:

.. code-block:: python

   class ScriptOptionalChainer(OptionalChainer):
       def should_instrument_file(self, filename: str) -> bool:
           return True

Then ``pyc bar.py -t pyccolo.examples.optional_chaining.ScriptOptionalChainer``
runs a whole script under the tracer (see :doc:`/reference/cli`).

Step 5: see the rewrite as source
---------------------------------

You don't have to *run* the code to use the augmentation. ``pyc.transform`` gives
you the desugared, plain-Python source, and ``pyc.untransform`` resugars it back:

.. code-block:: python

   with OptionalChainer:
       assert pyc.transform("y = a?.b?.c") == "y = a.b.c"
       tree = pyc.parse("y = a?.b?.c", instrument=False)
       assert pyc.untransform(tree) == "y = a?.b?.c"

This is the entry point for linters, formatters, and source maps — see
:doc:`/guides/source_to_source`.

Where to next
-------------

- :doc:`/concepts/model` explains how a token rewrite becomes a tagged
  AST node and then a handler call.
- The full example adds ``.?`` (permissive dereference), ``?.(`` (optional call),
  and ``??`` (nullish coalescing) — a good next read in
  ``pyccolo/examples/optional_chaining.py``.
