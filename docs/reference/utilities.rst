Utilities & globals
===================

.. currentmodule:: pyccolo

Smaller helpers and constants that round out the public surface. All are
importable from the top-level ``pyccolo`` package.

Tracer stack & contexts
-----------------------

.. autofunction:: pyccolo.tracer

.. autofunction:: pyccolo.instance

.. autofunction:: pyccolo.resolve_tracer

.. autofunction:: pyccolo.multi_context

.. autofunction:: pyccolo.tracing_context

.. autofunction:: pyccolo.tracing_enabled

.. autofunction:: pyccolo.tracing_disabled

.. autofunction:: pyccolo.instrumented

.. autofunction:: pyccolo.allow_reentrant_event_handling

Frames & guards
---------------

.. autofunction:: pyccolo.set_frame_local

.. autofunction:: pyccolo.make_guard_name

.. data:: pyccolo.PYCCOLO_BUILTIN_PREFIX

   The prefix used for the synthetic builtins Pyccolo injects (guard names,
   macro hooks). Handy when generating guard names or namespacing your own
   injected symbols.

Traceback visibility
--------------------

By default, frames from sandboxed ``exec``/``eval`` code are hidden from
tracebacks. These let you opt specific sandbox files back in.

.. autofunction:: pyccolo.mark_traceback_visible

.. autofunction:: pyccolo.is_traceback_visible

Sentinels
---------

The handler return-value sentinels are documented with the :doc:`handler
contract <handlers>`: :data:`~pyccolo.Null`, :data:`~pyccolo.Skip`,
:data:`~pyccolo.SkipAll`, and :data:`~pyccolo.Pass`.
