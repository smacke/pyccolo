Compose multiple tracers
========================

**Goal:** run several independently-written instrumentations at once.

Unlike hand-written :class:`ast.NodeTransformer` classes, Pyccolo tracers compose
by simply nesting their context managers. Return values thread through each
handler in order:

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
           assert env["x"] == 86   # (42 + 1) * 2

The same threading happens for multiple handlers **on one class** — they run in
definition order, each receiving the previous one's result as ``ret``.

Activating several at once
--------------------------

Rather than nesting many ``with`` blocks, activate a list of tracers together with
``pyc.multi_context``:

.. code-block:: python

   with pyc.multi_context([AddOne.instance(), TimesTwo.instance()]):
       env = pyc.exec("x = 42")
       assert env["x"] == 86

On the command line, pass multiple ``-t`` tracers to the ``pyc`` tool (see
:doc:`/reference/cli`); they compose the same way.

.. seealso::

   :doc:`/concepts/composition` explains *why* composition works — a single shared
   event-emission transform per node, rather than conflicting rewrites.
