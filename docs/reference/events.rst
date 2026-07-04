Event taxonomy
==============

Pyccolo emits over a hundred distinct *events*. Every one is a member of
:class:`pyccolo.trace_events.TraceEvent` and is also re-exported at the top level,
so ``pyc.before_stmt`` and ``TraceEvent.before_stmt`` are the same object. You
register a handler for an event with the :doc:`decorator or registration API
<registration>`, and the handler is invoked with the :doc:`standard handler
signature <handlers>`.

.. note::

   The table below is generated at build time directly from
   ``pyccolo/trace_events.py`` — the list of events, and the **thunk** / **sys** /
   **ast** flags, come straight from the source, so this page cannot drift from
   the library. If you are reading a released version, it describes exactly the
   events that version emits.

How to read the table
---------------------

- **Fires on** — the source construct (and moment) that triggers the event.
- **Handler return** — what the value your handler returns does. The common
  contracts are:

  - *replacement value* — the returned value **replaces** the value flowing out
    of the instrumented expression (return ``None`` to leave it unchanged; return
    :data:`pyccolo.Null` to force an actual ``None``). This is what powers
    behavior-changing tracers.
  - *callable (thunk)* — the event fires *before* an expression is evaluated, so
    the value in flight is a zero-argument callable that produces it. Return a
    (possibly different) callable to control whether and how it runs. These are
    exactly the events flagged **thunk** (they live in ``BEFORE_EXPR_EVENTS``).
  - *observe only* — the event is a notification; the return value is ignored
    (aside from the universal :data:`pyccolo.Skip` / :data:`pyccolo.SkipAll`
    controls).

- **Flags:**

  - **thunk** — a "before-expression" event; the handler receives/returns a
    thunk. See :doc:`handlers` for the calling convention.
  - **sys** — a native `sys.settrace
    <https://docs.python.org/3/library/sys.html#sys.settrace>`_ event; no AST
    rewriting is involved, so it fires without ``pyc.exec`` (see
    :doc:`/howto/sys_settrace`).
  - **ast:** *NodeType* — you may register a handler by passing this
    :mod:`ast` node type instead of the event, via ``AST_TO_EVENT_MAPPING``
    (e.g. ``@pyc.register_handler(ast.Assign)`` is ``after_assign_rhs``).

.. seealso::

   :doc:`handlers` for the exact arguments each handler receives, the sentinels
   (:data:`~pyccolo.Null`, :data:`~pyccolo.Skip`, :data:`~pyccolo.SkipAll`,
   :data:`~pyccolo.Pass`), and the thunk convention; :doc:`registration` for the
   decorator options (``when=``, ``guard=``, ``reentrant=``, ...).

The events
----------

.. pyccolo-events::

.. autoclass:: pyccolo.trace_events.TraceEvent
