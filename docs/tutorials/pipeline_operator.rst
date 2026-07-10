Tutorial: build a pipe operator (``|>``)
========================================

In this tutorial you'll add an Elixir/F#-style **pipe operator** to Python:
``x |> f`` means ``f(x)``, so a left-to-right chain like ``(1, 2, 3) |> list |>
sum`` reads in the order it runs. You'll see how a *binary-operator* augmentation
works, and how returning a function from ``before_binop`` lets you redefine what
an operator *does*. Then you'll add an assigning variant, ``|>>``, that stores the
running value in a variable.

This is the essence of the shipped ``pipeline_tracer.py`` example, minus the
placeholder-lambda machinery. Syntax augmentation requires Python ≥ 3.8.

Step 1: declare the operator token
----------------------------------

``|>`` isn't legal Python, so first we tell Pyccolo to rewrite it into a token the
parser *does* accept — the bitwise-or ``|`` — while remembering that this
particular ``|`` came from our spec. A ``binop``-position
:class:`~pyccolo.AugmentationSpec` does exactly that:

.. testcode::

   import pyccolo as pyc


   class Pipe(pyc.BaseTracer):
       global_guards_enabled = False       # pure syntax, no runtime guards

       pipe_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
       )

Setting ``global_guards_enabled = False`` marks the tracer as one that rewrites
syntax rather than emitting guarded runtime events, so it layers cleanly with
others (see :doc:`/guides/compose_tracers`).

Step 2: redefine what the operator does
---------------------------------------

After the rewrite, ``(1, 2, 3) |> list`` parses as an ordinary ``ast.BinOp`` whose
operator is ``ast.BitOr`` — but one Pyccolo has tagged as originating from our
spec. When such a binop is about to run, ``before_binop`` fires; returning a
two-argument function from it **replaces the operation itself**. We return one
that applies the right operand to the left:

.. testcode::

   class Pipe(pyc.BaseTracer):
       global_guards_enabled = False

       pipe_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
       )

       @pyc.before_binop(when=lambda node: isinstance(node.op, ast.BitOr))
       def apply(self, ret, node, *_, **__):
           if self.pipe_spec in self.get_augmentations(id(node)):
               return lambda left, right: right(left)   # x |> f  ==  f(x)
           return ret


   with Pipe:
       assert pyc.exec("out = (1, 2, 3) |> list")["out"] == [1, 2, 3]

Two things keep us precise. The ``when=`` guard skips the handler unless the
operator is a ``|`` at all, and
:meth:`~pyccolo.BaseTracer.get_augmentations` narrows us further to just the ``|``
tokens that came from ``|>`` — a real bitwise-or elsewhere in the program is left
completely alone.

Step 3: chain it
----------------

Because the handler runs for every ``|>`` node, chains compose for free — each
pipe feeds its result into the next:

.. testcode::

   with Pipe:
       env = pyc.exec("total = range(5) |> list |> sum")
   assert env["total"] == 10

Step 4: add an assigning pipe ``|>>``
-------------------------------------

A pipeline is more useful if you can tap the running value into a variable. We add
a second spec for ``|>>`` and handle it in the same place: instead of calling the
right operand, we *assign* the left one to the name on the right. (We pre-seed the
name to ``None`` first, so evaluating the right-hand operand doesn't raise
``NameError`` before our function runs.)

.. testcode::

   class Pipe(pyc.BaseTracer):
       global_guards_enabled = False

       pipe_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
       )
       assign_spec = pyc.AugmentationSpec(
           aug_type=pyc.AugmentationType.binop, token="|>>", replacement="|"
       )

       @pyc.before_binop(when=lambda node: isinstance(node.op, ast.BitOr))
       def apply(self, ret, node, frame, *_, **__):
           augmentations = self.get_augmentations(id(node))
           if self.assign_spec in augmentations:
               name = node.right                      # the variable to assign into
               frame.f_globals[name.id] = None        # avoid NameError on the RHS
               def stash(left, _right):
                   frame.f_globals[name.id] = left
                   return left                        # ...and keep piping it along
               return stash
           if self.pipe_spec in augmentations:
               return lambda left, right: right(left)
           return ret


   with Pipe:
       env = pyc.exec("(1, 2, 3) |> list |>> items\nresult = items |> sum")
   assert env["result"] == 6

``|>>`` returns the value it stashed, so the chain keeps flowing: we captured the
list into ``items`` mid-pipeline and then summed it.

Where to next
-------------

The shipped ``pipeline_tracer.py`` goes much further — placeholder lambdas
(``range(5) |> f[map(f[_ + 1], _)]``), tuple/dict-splat variants, and function
composition — and composes with the :doc:`quick-lambda tracer
</tutorials/quick_lambda>`. Both are built on the exact ``binop`` augmentation you
just wrote. To see the rewrite as plain source rather than running it, reach for
:doc:`/guides/source_to_source`.
