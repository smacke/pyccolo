Predicates
==========

.. currentmodule:: pyccolo

A predicate is the object behind a handler's ``when=`` option: it decides, per
node, whether the handler should fire. You can pass a plain callable of the node,
but wrapping it in a :class:`Predicate` gives you control over *when* the decision
is made (compile time vs. runtime) and lets you compose conditions. For the
task-oriented walkthrough, see :doc:`/guides/observe_and_override`.

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

A dynamic predicate re-checks each occurrence; here the handler fires only for
addition nodes, and only ``1 + 2`` matches:

.. testcode::

   fired = []


   class OnlyAdd(pyc.BaseTracer):
       @pyc.after_binop(when=pyc.Predicate(lambda node: isinstance(node.op, ast.Add)))
       def handle(self, ret, node, *_, **__):
           fired.append(type(node.op).__name__)
           return ret


   with OnlyAdd:
       pyc.exec("a = 1 + 2\nb = 3 * 4")
   assert fired == ["Add"]

Marking the same predicate ``static=True`` resolves it during the rewrite instead,
so a non-matching handler costs nothing at runtime. Use ``static`` whenever the
decision depends only on the shape of the syntax tree.

Composing predicates
--------------------

.. autoclass:: pyccolo.predicate.CompositePredicate
   :members:

:meth:`~pyccolo.predicate.CompositePredicate.any` /
:meth:`~pyccolo.predicate.CompositePredicate.all` combine predicates into one you
can pass as ``when=``. They short-circuit against :data:`Predicate.TRUE` /
:data:`Predicate.FALSE`, and the result stays static only if every input is
static:

.. testcode::

   from pyccolo.predicate import CompositePredicate

   is_add = pyc.Predicate(lambda node: isinstance(node.op, ast.Add), static=True)
   is_sub = pyc.Predicate(lambda node: isinstance(node.op, ast.Sub), static=True)
   add_or_sub = CompositePredicate.any([is_add, is_sub])

   seen = []


   class AddOrSub(pyc.BaseTracer):
       @pyc.register_handler(pyc.after_binop, when=add_or_sub)
       def handle(self, ret, node, *_, **__):
           seen.append(type(node.op).__name__)
           return ret


   with AddOrSub:
       pyc.exec("a = 1 + 2\nb = 5 - 1\nc = 2 * 3")
   assert seen == ["Add", "Sub"]

For a one-off combination you can also just write the disjunction inline —
``when=lambda node: isinstance(node.op, (ast.Add, ast.Sub))`` — and reach for
``CompositePredicate`` when you want to build predicates up programmatically or
preserve staticness across the combination.
