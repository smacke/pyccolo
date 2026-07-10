Compose multiple tracers
========================

The payoff of embedding instrumentation in source code is that independently
written tracers **layer without conflicting**. Where two hand-written
:class:`ast.NodeTransformer` classes would fight over how to rewrite the same
node, Pyccolo emits a single event per node and lets every interested handler
respond. You compose tracers by nesting their ``with`` blocks; their return
values thread through in order.

Nesting threads values through
------------------------------

Activate one tracer inside another and a value-carrying event visits each
handler in turn, innermost-last, each receiving the previous handler's result:

.. testcode::

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
   assert env["x"] == 86      # (42 + 1) * 2

Swap the nesting order and you get ``(42 * 2) + 1 == 85`` instead — composition is
ordered, like function application.

Activate several at once
------------------------

Nesting many ``with`` blocks gets unwieldy. :func:`pyccolo.multi_context`
activates a list of tracer instances together, with the same left-to-right
threading:

.. testcode::

   with pyc.multi_context([AddOne.instance(), TimesTwo.instance()]):
       assert pyc.exec("x = 42")["x"] == 86

On the command line, passing multiple ``-t`` tracers to the ``pyc`` tool composes
them the same way (see :doc:`/guides/tracing_real_programs`).

Multiple handlers on one tracer
-------------------------------

The same threading happens *within* a single class. Register more than one
handler for an event and they run in definition order, each handed the previous
result:

.. testcode::

   class TwoSteps(pyc.BaseTracer):
       @pyc.register_handler(pyc.after_assign_rhs)
       def add_one(self, ret, *_, **__):
           return ret + 1

       @pyc.register_handler(pyc.after_assign_rhs)
       def times_two(self, ret, *_, **__):
           return ret * 2


   with TwoSteps:
       assert pyc.exec("x = 42")["x"] == 86

The :data:`pyc.Skip` and :data:`pyc.SkipAll` sentinels (see
:doc:`/guides/observe_and_override`) let a handler cut this chain short — ``Skip``
stops the rest of *this* tracer's handlers, ``SkipAll`` stops every tracer below
it on the stack.

Tracers that only rewrite syntax
--------------------------------

Not every tracer emits runtime events. Some exist purely to *rewrite source* so
that other tracers can act on the result — the shipped ``brace_subscript.py``
example, for instance, swaps paired ``{`` / ``}`` for ``[`` / ``]`` so that
existing subscript-based macros accept brace syntax. A rewrite-only tracer
declares that it has no runtime footprint by setting
``global_guards_enabled = False``, which keeps it from adding per-node guards and
lets it layer cleanly under tracers that *do* emit events:

.. code-block:: python

   class BraceSubscriptTracer(pyc.BaseTracer):
       global_guards_enabled = False   # pure source rewrite, no runtime events
       # ...declares an AugmentationSpec, registers no handlers...

Because it contributes only a source transform, you can stack it beneath any
event-emitting tracer and the two never collide. See :doc:`/guides/add_syntax`
for how such rewrites are declared and :doc:`/concepts/composition` for *why* a
single shared event per node is what makes all of this compose.

Re-entrant handlers
-------------------

By default, Pyccolo does not re-enter a tracer's handlers for events that fire
*while a handler is already running* — this prevents runaway recursion when a
handler itself executes instrumented code. When you genuinely want a handler to
trace the code it triggers (a macro that expands into more instrumented code, for
example), opt in with the class attribute ``allow_reentrant_events = True``, or
register the individual handler with ``reentrant=True``. The bundled
``quasiquote.py`` and ``pipeline_tracer.py`` examples both rely on this. See
:doc:`/reference/config` for the class-level switch and
:doc:`/reference/registration` for the per-handler flag.
