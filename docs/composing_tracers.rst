Composing tracers
=================

A core feature of Pyccolo is that its instrumentation is *composable*. It is
usually tricky to use two or more :class:`ast.NodeTransformer` classes
simultaneously — sometimes you can have one inherit from the other, but if they
both define ``visit`` methods for the same AST node type you typically need a
bespoke node transformer that combines logic from each and resolves the corner
cases by hand.

With Pyccolo, you simply nest the context managers of each tracer class whose
instrumentation you wish to use, and everything usually Just Works:

.. code-block:: python

   import pyccolo as pyc


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
           assert env["x"] == 86  # (42 + 1) * 2 -- handlers compose in order

Return values compose across handlers on the same tracer as well as across
handlers on different tracers: the value returned by one handler becomes the
``ret`` argument of the next. The nesting order of the context managers
determines the order in which handlers run.

Why composition works
---------------------

Under the hood, every handler that cares about a given event shares a *single*
event-emission transform of the underlying AST node. Adding a second tracer does
not produce a second, conflicting rewrite of the same node — it simply
subscribes another handler to the event that the shared transform already emits.
This is what lets independently-written instrumentations layer without
interfering, and it is the same mechanism that makes rich syntax-augmentation
stacks (see :doc:`syntax_augmentation`) compose cleanly.

Composing more than two tracers at once is common enough that Pyccolo ships a
helper, :func:`pyccolo.multi_context`, which enters a list of context managers
together instead of hand-nesting ``with`` blocks.
