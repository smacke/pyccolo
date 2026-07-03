API reference
=============

This page is a compact reference for Pyccolo's public API. The narrative pages
under :doc:`Concepts <events_and_handlers>` are the best way to *learn* the
library; this appendix is for looking things up.

Everything documented here is importable from the top-level ``pyccolo`` package
(conventionally imported as ``pyc``).

Tracers
-------

.. autoclass:: pyccolo.BaseTracer
   :members:
   :undoc-members:

   .. automethod:: should_instrument_file

   .. automethod:: activate_guard

   .. automethod:: deactivate_guard

.. autoclass:: pyccolo.NoopTracer
   :members:

Registering handlers
--------------------

.. autofunction:: pyccolo.register_handler

.. autofunction:: pyccolo.register_raw_handler

.. autofunction:: pyccolo.skip_when_tracing_disabled

Every member of :class:`~pyccolo.trace_events.TraceEvent` is also re-exported as
a module-level decorator, so ``@pyc.before_stmt`` is sugar for
``register_handler(pyc.before_stmt)`` and accepts the same keyword arguments
(``when=``, ``reentrant=``, ...).

Running instrumented code
-------------------------

.. autofunction:: pyccolo.exec

.. autofunction:: pyccolo.eval

.. autofunction:: pyccolo.execute

.. autofunction:: pyccolo.parse

.. autofunction:: pyccolo.instrumented

.. autofunction:: pyccolo.tracer

.. autofunction:: pyccolo.instance

Tracing contexts
----------------

.. autofunction:: pyccolo.tracing_context

.. autofunction:: pyccolo.tracing_enabled

.. autofunction:: pyccolo.tracing_disabled

.. autofunction:: pyccolo.multi_context

Source-to-source
----------------

.. autofunction:: pyccolo.transform

.. autofunction:: pyccolo.untransform

.. autofunction:: pyccolo.is_pure_transform

Predicates
----------

.. autoclass:: pyccolo.Predicate
   :members:

.. autoclass:: pyccolo.predicate.CompositePredicate
   :members:

Syntax augmentation
-------------------

.. autoclass:: pyccolo.AugmentationSpec
   :members:

.. autoclass:: pyccolo.AugmentationType
   :members:
   :undoc-members:

.. autoclass:: pyccolo.CustomRewrite
   :members:

.. autoclass:: pyccolo.AstRewriter
   :members:

State
-----

.. autoclass:: pyccolo.TraceStack
   :members:

Return-value sentinels
----------------------

Handlers use these sentinels to express intents that a bare ``return`` cannot
(see :doc:`events_and_handlers`):

- ``pyccolo.Null`` — override the instrumented value *with* ``None`` (as opposed
  to "no override", which is what returning ``None`` means).
- ``pyccolo.Skip`` — stop running further handlers for the current event on this
  tracer.
- ``pyccolo.SkipAll`` — abort the entire tracer stack for the current event.
- ``pyccolo.Pass`` — an explicit "do nothing / no override" marker.

The event taxonomy
------------------

.. autoclass:: pyccolo.trace_events.TraceEvent

All events are members of :class:`pyccolo.trace_events.TraceEvent`, each
re-exported as ``pyc.<name>``. The complete list (100+ events) is defined in
`pyccolo/trace_events.py
<https://github.com/smacke/pyccolo/blob/master/pyccolo/trace_events.py>`_.
Grouped by what they instrument:

- **Statements:** ``before_stmt``, ``after_stmt``, ``after_module_stmt``,
  ``after_expr_stmt``.
- **Names & assignments:** ``load_name``, ``after_assign_rhs``,
  ``before_augassign_rhs`` / ``after_augassign_rhs``.
- **Attributes:** ``before_attribute_load`` / ``after_attribute_load``,
  ``before_attribute_store``, ``before_attribute_del``.
- **Subscripts:** ``before_subscript_load`` / ``after_subscript_load``,
  ``before_subscript_store``, ``before_subscript_del``, ``_load_saved_slice``.
- **Literals:** ``after_int``, ``after_float``, ``after_complex``,
  ``after_string``, ``after_bytes``, ``after_list_literal``,
  ``after_dict_literal``, ``after_set_literal``, ``after_tuple_literal``,
  ``ellipsis``.
- **Operators:** ``before_binop`` / ``after_binop``, ``before_unaryop`` /
  ``after_unaryop``, ``before_boolop`` / ``after_boolop``, ``before_compare`` /
  ``after_compare``, and the ``*_arg`` variants.
- **Comprehensions & calls:** the comprehension events, ``before_call`` /
  ``after_call``, and the argument events.
- **Control flow:** the loop events (``before_for_loop_body``,
  ``after_for_loop_iter``, ``after_while_loop_iter``, ...), ``after_if_test``,
  ``after_while_test``.
- **Functions:** ``before_function_body``, ``after_function_execution``,
  ``before_lambda`` / ``after_lambda_body``.
- **Imports & module lifecycle:** ``before_import``, ``init_module``,
  ``exit_module``.
- **``sys``-level events** (see :doc:`sys_settrace`): ``call``, ``line``,
  ``return_``, ``exception``, ``opcode``.
