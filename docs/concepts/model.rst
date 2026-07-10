The model: events and handlers
==============================

Pyccolo has exactly two moving parts, and once they click the rest of the library
follows. You write **handlers** — ordinary methods on a tracer class — and you
register each one for an **event**. Pyccolo arranges for the event to fire at the
right moment while your program runs, and calls your handler.

That is the whole mental model. Everything else — syntax augmentation,
source-to-source, guards — is built on top of it.

Events
------

An event marks a specific, meaningful moment during execution: *a statement is
about to run*, *a float literal was just evaluated*, *an attribute is being
loaded*, *a stack frame was pushed*. Pyccolo exposes a fine-grained taxonomy of
over a hundred of them, spanning:

- **statements** — before and after each statement runs;
- **names and literals** — loading a variable, evaluating an ``int``, ``float``,
  ``str``, ``bytes``, or collection literal;
- **operators** — binary, unary, boolean, and comparison operations (and their
  individual operands);
- **attributes and subscripts** — ``obj.attr`` and ``obj[key]`` in load, store,
  and delete contexts;
- **calls and control flow** — calls, arguments, returns, loops, ``if`` / ``while``
  tests, comprehensions;
- **imports and module lifecycle** — module entry and exit, ``import`` statements;
- **``sys`` events** — ``call``, ``line``, ``return_``, ``exception``, and
  ``opcode``, which ride on Python's built-in `sys.settrace
  <https://docs.python.org/3/library/sys.html#sys.settrace>`_ rather than on AST
  rewriting.

Every event is a member of the ``TraceEvent`` enumeration and is also re-exported
at the top level, so ``pyc.before_stmt`` and ``TraceEvent.before_stmt`` are the
same object. The full, code-derived catalog — including which events let a handler
override a value and which are "before-expression" (thunk) events — lives in the
:doc:`/reference/events` reference.

Handlers
--------

A handler is a method you decorate with the event it should run for:

.. code-block:: python

   import pyccolo as pyc


   class HelloTracer(pyc.BaseTracer):
       @pyc.before_stmt
       def handle_stmt(self, *_, **__):
           print("Hello, world!")

Every handler is called with the same four positional arguments:

1. ``ret`` — the value currently flowing through the instrumented expression;
2. ``node`` — the AST node being instrumented (or its integer id, for raw
   handlers, or ``None`` for ``sys`` events);
3. ``frame`` — the stack frame at the instrumentation point;
4. ``event`` — the event that fired, useful when one handler serves several
   events.

Because most handlers only care about one or two of these, the idiomatic shape
accepts the first argument it needs and swallows the rest::

   def handle(self, ret, *_, **__):
       ...

Some events pass extra keyword arguments (a ``guard`` name, an ``is_last`` flag,
and so on); the ``**__`` keeps your handler forward-compatible as those grow. The
precise contract — arguments, keyword arguments, and return-value sentinels — is
in :doc:`/reference/handlers`.

Observe, or override
--------------------

Handlers are not limited to watching. For value-carrying events, **the value a
handler returns replaces the value of the instrumented expression**. That single
rule is what lets an entire behavioral change fit inside one handler — for
example, promoting every ``float`` literal to :class:`~decimal.Decimal`:

.. code-block:: python

   from decimal import Decimal
   import pyccolo as pyc


   class ExactFloats(pyc.BaseTracer):
       @pyc.after_float
       def to_decimal(self, ret, *_, **__):
           return Decimal(str(ret))


   with ExactFloats:
       pyc.exec("print(0.1 + 0.2)")  # -> 0.3

Returning ``None`` (or nothing) means "don't override"; returning the ``pyc.Null``
sentinel overrides *with* a real ``None``. See :doc:`/guides/observe_and_override` for
the full set of overriding moves.

.. _pay-as-you-go:

Pay-as-you-go instrumentation
-----------------------------

A hundred events sounds expensive, but you never pay for events you do not use.
For AST-level events, Pyccolo transforms your source to emit an event **only when
at least one active tracer has a handler registered for it**. If nobody is
listening for ``after_binop``, no binary operation in your code is touched, and
the rewritten source stays lean. Adding a handler is what "turns on" the
corresponding rewrite.

Pay-as-you-go decides whether an event is emitted *at all*. To go finer and gate
an *individual* handler to only the nodes it cares about, give it a
:ref:`when predicate <when-predicate>`. A ``when`` predicate that depends only on
the shape of the syntax tree can be marked **static**, in which case it too is
resolved at instrument time — so an uninterested handler adds nothing at runtime,
in the same spirit as pay-as-you-go itself.

This is also why instrumentation composes so well: many handlers can subscribe to
the same emitted event without competing to rewrite the same node — the reason
multiple tracers layer cleanly is :doc:`/concepts/composition`. The rest of this
page follows a snippet of source through the transform that makes all of this
happen.

From source to running program
==============================

You do not need any of what follows to *use* Pyccolo — the events-and-handlers
model above is enough — but understanding the pipeline makes the more advanced
features (syntax augmentation, guards, source maps) feel inevitable rather than
magical. A snippet travels through six stages on its way from text to a running,
instrumented program.

1. Syntax augmentation (optional)
---------------------------------

If any active tracer declares new surface syntax, the raw source is first rewritten
at the *token* level: otherwise-illegal spans like ``?.`` or ``|>`` are replaced
with legal ones (``.`` and ``|`` respectively) so that Python's own parser will
accept the text. Crucially, Pyccolo **remembers where each rewrite happened**, so
a later handler can ask "did this node come from a ``?.`` token?" and behave
accordingly.

If no tracer augments syntax, this stage does nothing and the source is already
valid Python. Augmentation is covered in :doc:`/guides/add_syntax` and
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
this is the pay-as-you-go property from above. Because a single shared emission
point is installed per node, any number of handlers can subscribe without
competing rewrites; that is the root of Pyccolo's
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
amortize instrumentation overhead; see :doc:`/guides/guards_and_performance` and
:meth:`~pyccolo.BaseTracer.activate_guard`.

What about ``sys`` events?
--------------------------

The ``call``, ``line``, ``return_``, ``exception``, and ``opcode`` events skip
stages 1–4 entirely. They ride on Python's built-in `sys.settrace
<https://docs.python.org/3/library/sys.html#sys.settrace>`_ machinery, so there is
no AST rewriting involved — which is why handlers for them work without running
your code through ``pyc.exec`` (see :doc:`/guides/tracing_real_programs` and
:doc:`/concepts/exec_and_scoping`).

Getting the source instead of running it
----------------------------------------

Stages 1–2 are also useful on their own. ``pyc.transform`` stops after the rewrite
and hands you the transformed *source*; ``pyc.untransform`` reverses an
augmentation, resugaring valid Python back into the augmented syntax. Both can
remap source positions for source-map tooling. See :doc:`/guides/source_to_source`
and :doc:`/reference/source_api`.
