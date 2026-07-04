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
sentinel overrides *with* a real ``None``. See :doc:`/howto/override_values` for
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
the same emitted event without competing to rewrite the same node. The mechanics
of that transform are the subject of :doc:`/concepts/rewrite_pipeline`, and the
reason multiple tracers layer cleanly is :doc:`/concepts/composition`.
