Registering handlers
====================

There are three equivalent ways to attach a :doc:`handler <handlers>` to an
:doc:`event <events>`. All accept the same options.

The event decorator
-------------------

Every event is re-exported as a decorator, so the shortest form is to decorate a
method with the event itself:

.. code-block:: python

   class HelloTracer(pyc.BaseTracer):
       @pyc.before_stmt
       def handle(self, *_, **__):
           print("Hello, world!")

This is exactly sugar for ``register_handler(pyc.before_stmt)`` and accepts the
same keyword arguments:

.. code-block:: python

   @pyc.before_attribute_load(when=lambda node: node.attr == "secret", reentrant=True)
   def handle(self, obj, node, *_, **__):
       ...

``register_handler``
--------------------

.. autofunction:: pyccolo.register_handler

The ``event`` argument may be a single :class:`~pyccolo.trace_events.TraceEvent`,
an :class:`ast.AST` **subclass** (resolved through ``AST_TO_EVENT_MAPPING`` — e.g.
``ast.Assign`` means ``after_assign_rhs``), or a tuple of either to register one
handler for several events at once:

.. code-block:: python

   @pyc.register_handler((pyc.after_for_loop_iter, pyc.after_while_loop_iter))
   def after_loop_iter(self, *_, guard, **__):
       self.activate_guard(guard)

Keyword options:

============================  ==================================================
Option                        Effect
============================  ==================================================
``when``                      A predicate gating the handler — a plain callable
                              of the node, or a :doc:`Predicate <predicates>`.
                              Documented in detail :ref:`below <when-predicate>`.
``reentrant``                 Allow the handler to fire while an event is already
                              being handled (see ``allow_reentrant_events``).
``static``                    Evaluate ``when`` once at instrument time only.
``use_raw_node_id``           Pass the integer node id instead of the resolved
                              :class:`ast.AST` node (see
                              :func:`register_raw_handler`).
``guard``                     Associate a guard with this handler so it can be
                              switched off at runtime; see
                              :doc:`/howto/performance_guards`.
``exempt_from_guards``        Keep firing even when the enclosing guard is active.
============================  ==================================================

.. _when-predicate:

Gating a handler with ``when``
------------------------------

``when`` decides, **per node**, whether this handler should fire at all. Where
:ref:`pay-as-you-go instrumentation <pay-as-you-go>` chooses whether an event is
emitted *at all* (based on whether *any* active tracer listens for it), ``when``
narrows a *specific* handler down to the *specific* nodes it cares about — for
example "only ``.load`` accesses whose attribute is ``secret``", or "only
``or``-shaped boolean ops".

What you can pass
~~~~~~~~~~~~~~~~~

``when`` accepts either of:

- **A plain callable** — receives the node and returns a ``bool``. By default the
  node is the resolved :class:`ast.AST`; with ``use_raw_node_id=True`` it is the
  integer node id instead (use this when you only need identity or intend to look
  the node up yourself). Returning ``True`` lets the handler fire; ``False``
  suppresses it.

  .. code-block:: python

     @pyc.before_attribute_load(when=lambda node: node.attr == "secret")
     def redact(self, obj, node, *_, **__):
         ...

- **A** :doc:`Predicate <predicates>` — the same idea, but an object that also
  carries *when* the decision is made (see below) and can be composed. A plain
  callable is wrapped into a ``Predicate`` for you.

Static vs. dynamic evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is the important distinction, and it is what ``static`` controls:

- **Static** (``static=True``, or a ``Predicate(..., static=True)``) — the
  predicate depends only on the *shape* of the tree, so it is evaluated **once,
  at instrument time**, and the answer is baked into the rewrite. A static
  predicate therefore costs **nothing at runtime**: nodes it rejects are simply
  never wired to your handler. Reach for this whenever the condition is a pure
  function of the AST (an ``isinstance`` check on ``node.op``, an attribute name,
  a node's position in the tree via :meth:`~pyccolo.BaseTracer.is_outer_stmt`,
  ...).

  .. code-block:: python

     import ast

     @pyc.before_boolop(when=lambda node: isinstance(node.op, ast.Or), static=True)
     def only_or(self, ret, node, *_, **__):
         ...

- **Dynamic** (the default) — the predicate is re-evaluated on **every** event,
  via :meth:`Predicate.dynamic_call <pyccolo.Predicate.dynamic_call>`. Use this
  only when the decision depends on runtime state the tree cannot tell you about;
  it is strictly more expensive than a static predicate.

If you pass a bare callable without ``static=True``, it is treated as dynamic. If
you pass a ``Predicate``, its own ``.static`` flag wins.

Composing predicates
~~~~~~~~~~~~~~~~~~~~~

Combine conditions with :meth:`CompositePredicate.any
<pyccolo.predicate.CompositePredicate.any>` /
:meth:`~pyccolo.predicate.CompositePredicate.all`, and short-circuit against the
:data:`Predicate.TRUE <pyccolo.Predicate.TRUE>` /
:data:`~pyccolo.Predicate.FALSE` constants. A composite stays static only if
*every* constituent is static, so mixing in one dynamic predicate makes the whole
check dynamic.

.. seealso::

   :doc:`/howto/conditional_handlers` for a task-oriented walkthrough, and
   :doc:`/reference/predicates` for the full :class:`~pyccolo.Predicate` API.

``register_raw_handler``
------------------------

.. autofunction:: pyccolo.register_raw_handler

Identical to :func:`register_handler` with ``use_raw_node_id=True``: the handler
receives the raw integer node id rather than a resolved :class:`ast.AST` node.
This is the fastest path when you only need node **identity** (e.g. to look
values up in a dict keyed by ``id(node)``) and want to avoid resolving the node:

.. code-block:: python

   @pyc.register_raw_handler(pyc.before_stmt)
   def handle_stmt(self, _ret, stmt_id, frame, *_, **__):
       if stmt_id not in self.seen_stmts:
           self.seen_stmts.add(stmt_id)

``skip_when_tracing_disabled``
------------------------------

.. autofunction:: pyccolo.skip_when_tracing_disabled
