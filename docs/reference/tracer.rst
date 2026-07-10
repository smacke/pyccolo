``BaseTracer``
==============

.. currentmodule:: pyccolo

Everything you build with Pyccolo is a subclass of :class:`BaseTracer`. Thanks to
its metaclass, **the class itself acts as the tracer** for most purposes: it is a
singleton, it doubles as a context manager, and class-level calls are forwarded to
the instance.

.. autoclass:: pyccolo.BaseTracer

Activation & singleton
----------------------

A tracer class is a `traitlets <https://traitlets.readthedocs.io/>`_ singleton.
You rarely instantiate it yourself:

- ``MyTracer.instance()`` — the singleton instance (also ``pyc.instance()`` for
  the active tracer).
- ``with MyTracer:`` — activate for a dynamic scope (the class is a context
  manager). Equivalent to ``with MyTracer.instance().tracing_context():``.
- ``MyTracer.enable()`` / ``MyTracer.disable()`` — activate/deactivate imperatively.

.. rubric:: Context managers

.. automethod:: pyccolo.BaseTracer.tracing_context
.. automethod:: pyccolo.BaseTracer.tracing_enabled
.. automethod:: pyccolo.BaseTracer.tracing_disabled
.. automethod:: pyccolo.BaseTracer.tracing_non_context

Running & compiling code
------------------------

See :doc:`/concepts/exec_and_scoping` for *why* code defined in the same file as
the tracer must be run through these entry points.

.. automethod:: pyccolo.BaseTracer.exec
.. automethod:: pyccolo.BaseTracer.eval
.. automethod:: pyccolo.BaseTracer.execute
.. automethod:: pyccolo.BaseTracer.parse
.. automethod:: pyccolo.BaseTracer.transform
.. automethod:: pyccolo.BaseTracer.untransform
.. automethod:: pyccolo.BaseTracer.instrumented
.. automethod:: pyccolo.BaseTracer.trace_lambda

.. rubric:: Re-entrant fragments

For instrumenting code from *inside* a handler without disturbing tracer state
(used by macro-style tracers):

.. automethod:: pyccolo.BaseTracer.parse_fragment
.. automethod:: pyccolo.BaseTracer.exec_fragment

Overridable hooks
-----------------

Override these on your subclass to customize behavior. All have safe defaults.

.. automethod:: pyccolo.BaseTracer.should_instrument_file
.. automethod:: pyccolo.BaseTracer.file_passes_filter_for_event
.. automethod:: pyccolo.BaseTracer.enter_tracing_hook
.. automethod:: pyccolo.BaseTracer.exit_tracing_hook
.. automethod:: pyccolo.BaseTracer.static_init_module
.. automethod:: pyccolo.BaseTracer.should_propagate_handler_exception

Guards
------

Guards let you switch instrumentation off at runtime for a function or loop once
you no longer need it — the key to :doc:`amortizing overhead
</guides/guards_and_performance>`.

.. automethod:: pyccolo.BaseTracer.activate_guard
.. automethod:: pyccolo.BaseTracer.deactivate_guard
.. automethod:: pyccolo.BaseTracer.register_local_guard

Augmentation helpers
--------------------

Used when defining :doc:`new syntax </guides/add_syntax>`.

.. automethod:: pyccolo.BaseTracer.syntax_augmentation_specs
.. automethod:: pyccolo.BaseTracer.get_augmentations
.. automethod:: pyccolo.BaseTracer.make_ast_rewriter
.. automethod:: pyccolo.BaseTracer.make_syntax_augmenter
.. automethod:: pyccolo.BaseTracer.make_sandbox_fname

AST ancestry
------------

Classmethods for asking where a node sits in the tree — handy inside ``when=``
predicates and handlers.

.. automethod:: pyccolo.BaseTracer.is_outer_stmt
.. automethod:: pyccolo.BaseTracer.is_initial_frame_stmt
.. automethod:: pyccolo.BaseTracer.stmt_only_has_ancestor_types

State
-----

.. automethod:: pyccolo.BaseTracer.make_stack
.. automethod:: pyccolo.BaseTracer.reset

``NoopTracer``
--------------

.. autoclass:: pyccolo.NoopTracer

The default tracer used when the stack is empty: it patches nothing, enables no
guards, and instruments no files.
