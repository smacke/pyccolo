Conditional handlers and predicates
===================================

Sometimes a handler should only fire for *some* of the nodes that emit an event
— additions but not subtractions, a particular variable name, and so on. Every
event decorator accepts a ``when=`` keyword for exactly this:

.. code-block:: python

   import ast

   import pyccolo as pyc


   class TraceAdditions(pyc.BaseTracer):
       @pyc.after_binop(when=lambda node: isinstance(node.op, ast.Add))
       def handle_add(self, ret, node, *_, **__):
           print(f"added -> {ret}")


   with TraceAdditions:
       pyc.exec("x = 1 + 2\ny = 3 * 4")  # only the addition is reported

The ``when=`` callable receives the AST node for the instrumented construct and
returns a truthy/falsy value. Because it is handed the node, it can inspect
operator type, identifier names (``node.id``), literal values, and anything else
in the tree.

The ``Predicate`` class
-----------------------

Under the hood, ``when=`` values are wrapped in :class:`pyccolo.Predicate`. You
can also pass a ``Predicate`` directly, which unlocks two extra knobs:

- **static vs. dynamic.** ``Predicate(condition, static=True)`` is evaluated once
  at *instrumentation (compile) time* — if it is false, no instrumentation is
  emitted for that node at all, so there is zero runtime cost. A dynamic
  predicate (the default for raw-node-id predicates) is re-checked at *runtime*
  each time the node executes, which is necessary when the decision depends on
  runtime state.
- **raw node ids.** ``Predicate(condition, use_raw_node_id=True)`` passes the
  node's integer id rather than the node object, mirroring
  :func:`pyccolo.register_raw_handler`.

The sentinels ``Predicate.TRUE`` and ``Predicate.FALSE`` are always-on and
always-off predicates, handy as building blocks.

Composing predicates
--------------------

:class:`pyccolo.predicate.CompositePredicate` combines several predicates with a
reducer. The classmethods ``CompositePredicate.any`` and
``CompositePredicate.all`` build the ``or`` / ``and`` combinations and
short-circuit against the ``TRUE`` / ``FALSE`` sentinels:

.. code-block:: python

   from pyccolo.predicate import CompositePredicate, Predicate

   # true if *either* base predicate is true
   either = CompositePredicate.any([pred_a, pred_b])

   # CompositePredicate.any([Predicate.TRUE, ...]) collapses to Predicate.TRUE
   # CompositePredicate.all([]) is Predicate.TRUE, and any([]) is TRUE as well

A composite is considered static only when *all* of its constituents are static;
if any constituent is dynamic, the composite is re-checked at runtime.

For worked examples, see
`test/test_predicate.py <https://github.com/smacke/pyccolo/blob/master/test/test_predicate.py>`_
and the ``when=`` usages in
`test/test_handlers.py <https://github.com/smacke/pyccolo/blob/master/test/test_handlers.py>`_.
