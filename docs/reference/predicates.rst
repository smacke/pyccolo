Predicates
==========

.. currentmodule:: pyccolo

A predicate is the object behind a handler's ``when=`` option: it decides, per
node, whether the handler should fire. You can pass a plain callable of the node,
but wrapping it in a :class:`Predicate` gives you control over *when* the decision
is made (compile time vs. runtime) and lets you compose conditions.

.. autoclass:: pyccolo.Predicate
   :members:

- ``static=True`` predicates are evaluated **once, at instrument time**, and the
  result is baked into the rewrite — zero runtime cost, but the node must be
  decidable statically.
- ``static=False`` predicates re-run on every event (via
  :meth:`~Predicate.dynamic_call`).
- ``use_raw_node_id=True`` passes the integer node id to your condition instead of
  the resolved :class:`ast.AST` node.
- :data:`Predicate.TRUE` / :data:`Predicate.FALSE` are ready-made constants.

Composing predicates
--------------------

.. autoclass:: pyccolo.predicate.CompositePredicate
   :members:

Use :meth:`~pyccolo.predicate.CompositePredicate.any` /
:meth:`~pyccolo.predicate.CompositePredicate.all` to combine predicates; they
short-circuit against :data:`Predicate.TRUE` / :data:`Predicate.FALSE` and stay
static only if every input is static.

.. seealso::

   :doc:`/howto/conditional_handlers` for worked examples.
