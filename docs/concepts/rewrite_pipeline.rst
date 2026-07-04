How instrumentation works
=========================

This page follows a snippet of source all the way from text to a running,
instrumented program. You do not need any of it to *use* Pyccolo — the
:doc:`/concepts/model` is enough — but understanding the pipeline makes the more
advanced features (syntax augmentation, guards, source maps) feel inevitable
rather than magical.

The pipeline has six stages.

1. Syntax augmentation (optional)
---------------------------------

If any active tracer declares new surface syntax, the raw source is first rewritten
at the *token* level: otherwise-illegal spans like ``?.`` or ``|>`` are replaced
with legal ones (``.`` and ``|`` respectively) so that Python's own parser will
accept the text. Crucially, Pyccolo **remembers where each rewrite happened**, so
a later handler can ask "did this node come from a ``?.`` token?" and behave
accordingly.

If no tracer augments syntax, this stage does nothing and the source is already
valid Python. Augmentation is covered in :doc:`/howto/add_syntax` and
:doc:`/reference/augmentation`.

2. Parse
--------

The (now legal) source is parsed into a standard :mod:`ast` tree. From here on,
Pyccolo works with ordinary Python AST nodes.

3. Rewrite: weave in event emission
-----------------------------------

An ``AstRewriter`` walks the tree. For each node whose event *some active tracer
has a handler for*, it wraps the node so that, at runtime, evaluating it also
emits the corresponding event. A node nobody is listening to is left untouched —
this is the "pay-as-you-go" property from :doc:`/concepts/model`. Because a single
shared emission point is installed per node, any number of handlers can subscribe
without competing rewrites; that is the root of Pyccolo's
:doc:`composability </concepts/composition>`.

4. Bookkeeping
--------------

As it rewrites, Pyccolo builds bookkeeping tables. Every AST node is keyed by its
integer id, so the runtime can hand a handler either the resolved node or just its
id (the fast path — see :doc:`/reference/registration`). It also records
parent / containing relationships and, where relevant, which augmentations
produced a node. These tables are what power node-ancestry queries like
:meth:`~pyccolo.BaseTracer.is_outer_stmt` and the
:meth:`~pyccolo.BaseTracer.get_augmentations` lookups a syntax-defining handler
relies on. (A tracer that needs none of this can set
``requires_ast_bookkeeping = False``; see :doc:`/reference/config`.)

5. Run: emit events, dispatch handlers
--------------------------------------

The rewritten tree is compiled and executed. Each woven-in emission point calls
into Pyccolo's event machinery, which walks the **active tracer stack** and runs
every handler registered for that event. Return values are *threaded*: each
handler's result becomes the ``ret`` passed to the next, so overrides accumulate
in a well-defined order across handlers on one tracer and across the whole stack.
Handlers can also short-circuit the chain with the ``pyc.Skip`` /
``pyc.SkipAll`` sentinels (see :doc:`/reference/handlers`).

Instrumented code run through ``pyc.exec`` / ``pyc.eval`` is compiled under a
synthetic **sandbox filename** (``pyc.SANDBOX_FNAME`` and the shared
``pyc.SANDBOX_FNAME_PREFIX``), so its frames are recognizable in tracebacks and can
be filtered by handlers — the coverage example, for instance, ignores frames whose
filename starts with that prefix.

6. Guards: turn instrumentation back off
----------------------------------------

Emission points are gated by **guards** — runtime booleans associated with a
function or loop. When a tracer decides it no longer needs to watch a particular
region (say, after a loop's first iteration), it activates the guard and
subsequent runs execute at native speed with no event emission. This is how you
amortize instrumentation overhead; see :doc:`/howto/performance_guards` and
:meth:`~pyccolo.BaseTracer.activate_guard`.

What about ``sys`` events?
--------------------------

The ``call``, ``line``, ``return_``, ``exception``, and ``opcode`` events skip
stages 1–4 entirely. They ride on Python's built-in `sys.settrace
<https://docs.python.org/3/library/sys.html#sys.settrace>`_ machinery, so there is
no AST rewriting involved — which is why handlers for them work without running
your code through ``pyc.exec`` (see :doc:`/howto/sys_settrace` and
:doc:`/concepts/exec_and_scoping`).

Getting the source instead of running it
----------------------------------------

Stages 1–2 are also useful on their own. ``pyc.transform`` stops after the rewrite
and hands you the transformed *source*; ``pyc.untransform`` reverses an
augmentation, resugaring valid Python back into the augmented syntax. Both can
remap source positions for source-map tooling. See :doc:`/howto/source_to_source`
and :doc:`/reference/source_api`.
