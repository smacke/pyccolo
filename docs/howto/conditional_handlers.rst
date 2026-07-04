Fire a handler only sometimes
=============================

**Goal:** run a handler for *some* nodes, not every occurrence of an event.

Pass a ``when=`` predicate when registering the handler. In its simplest form it
is a plain callable of the AST node:

.. code-block:: python

   import ast
   import pyccolo as pyc

   class OnlyOrBoolops(pyc.BaseTracer):
       @pyc.before_boolop(when=lambda node: isinstance(node.op, ast.Or))
       def handle(self, ret, node, *_, **__):
           ...

Static vs. dynamic predicates
-----------------------------

Wrapping the condition in a ``Predicate`` gives you control over *when* it is
evaluated:

- ``Predicate(cond, static=True)`` is evaluated **once, at instrument time**, and
  the result is baked into the rewrite — zero runtime cost, but the node must be
  decidable statically.
- ``Predicate(cond)`` (dynamic) re-runs on every event.
- ``Predicate(cond, use_raw_node_id=True)`` passes the integer node id to your
  condition instead of the resolved node.

``Predicate.TRUE`` and ``Predicate.FALSE`` are ready-made constants.

Combining conditions
--------------------

Use ``CompositePredicate.any`` / ``CompositePredicate.all`` to combine predicates;
they short-circuit against ``Predicate.TRUE`` / ``Predicate.FALSE`` and stay
static only if every input is static:

.. code-block:: python

   from pyccolo.predicate import CompositePredicate
   from pyccolo import Predicate

   pred = CompositePredicate.any([Predicate(is_foo), Predicate(is_bar)])

.. seealso::

   :doc:`/reference/predicates` for the full API and :doc:`/reference/registration`
   for the other handler options (``guard=``, ``reentrant=``, ``use_raw_node_id=``).
