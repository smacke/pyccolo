Why composing tracers works
===========================

One of Pyccolo's defining features is that independently-written instrumentations
usually layer together without any special effort. This page explains why — and
why the same task is painful with hand-written AST transformers.

The usual pain
--------------

Suppose you have two separate analyses, each written as an
:class:`ast.NodeTransformer`. If they both define a ``visit_BinOp`` (or otherwise
touch the same node type), you cannot simply run one and then the other: the second
transformer sees the tree the first already rewrote, and their edits collide.
Sometimes you can make one subclass the other, but in general you end up writing a
third, bespoke transformer that manually interleaves the logic of both and resolves
every corner case by hand. Every new combination is new code.

The Pyccolo approach
--------------------

Pyccolo sidesteps this entirely. Instead of each tracer rewriting the tree its own
way, there is a **single, shared event-emission transform** installed per node (see
:doc:`/concepts/model`). Every handler that cares about that node — no
matter which tracer it belongs to — simply subscribes to the same emitted event.
There are no competing rewrites to reconcile, because there is only ever one
rewrite.

Activating several tracers is then just nesting their contexts:

.. code-block:: python

   class AddOne(pyc.BaseTracer):
       @pyc.after_assign_rhs
       def handle(self, ret, *_, **__):
           return ret + 1


   class TimesTwo(pyc.BaseTracer):
       @pyc.after_assign_rhs
       def handle(self, ret, *_, **__):
           return ret * 2


   with AddOne:
       with TimesTwo:
           env = pyc.exec("x = 42")
           assert env["x"] == 86  # (42 + 1) * 2

(You can also activate several at once with ``pyc.multi_context``, or pass multiple
``-t`` tracers to the :doc:`pyc CLI </reference/cli>`.)

How return values compose
-------------------------

When more than one handler fires for the same event, their return values are
**threaded** in order: the first handler's result becomes the ``ret`` argument of
the second, and so on. Above, ``AddOne`` runs first and turns ``42`` into ``43``;
``TimesTwo`` then receives ``43`` and turns it into ``86``. This threading works
identically whether the handlers live on one tracer (they run in definition order)
or across the whole tracer stack (outermost context first). The step-by-step
recipe is in :doc:`/guides/compose_tracers`.

Transformational tracers
------------------------

A tracer that *only* rewrites syntax — rather than observing runtime values —
typically sets ``global_guards_enabled = False`` (see :doc:`/reference/config`).
This keeps its transform purely structural so that it layers cleanly under or over
value-observing tracers without the guard machinery getting in the way. The
optional-chaining and pipeline examples are built this way.
