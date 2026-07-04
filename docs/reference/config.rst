Configuration attributes
=========================

Behavior that applies to a whole tracer is set with **class-level attributes** on
your :class:`~pyccolo.BaseTracer` subclass. Set them in the class body:

.. code-block:: python

   class CoverageTracer(pyc.BaseTracer):
       bytecode_caching_allowed = False
       # ... handlers ...

.. list-table::
   :header-rows: 1
   :widths: 26 12 62

   * - Attribute
     - Default
     - Effect
   * - ``instrument_all_files``
     - ``False``
     - Instrument every imported file, ignoring
       :meth:`~pyccolo.BaseTracer.should_instrument_file`.
   * - ``allow_reentrant_events``
     - ``False``
     - Allow handlers to fire while another event is already being handled.
       Needed by macro-style tracers (see ``quasiquote``); can also be entered
       locally with :func:`pyccolo.allow_reentrant_event_handling`.
   * - ``multiple_threads_allowed``
     - ``False``
     - Emit events from threads other than the one that entered the tracing
       context.
   * - ``requires_ast_bookkeeping``
     - ``True``
     - Retain the AST bookkeeping tables (node-by-id, parent links,
       augmentations). Turn off only if no handler needs resolved nodes.
   * - ``should_patch_meta_path``
     - ``True``
     - Install the ``sys.meta_path`` import hook when tracing, so imports can be
       instrumented.
   * - ``global_guards_enabled``
     - ``True``
     - Enable the global guard machinery. Purely *transformational* tracers (that
       only rewrite syntax) set this ``False`` so they compose cleanly.
   * - ``bytecode_caching_allowed``
     - ``True``
     - Allow cached instrumented bytecode. Coverage-style tracers set this
       ``False`` to avoid stale/omitted statements.
   * - ``instrument_lambdas``
     - ``False``
     - Let :meth:`~pyccolo.BaseTracer.instrumented` weave bare ``lambda``\\ s
       (lifted into synthetic ``def``\\ s).
   * - ``keep_sandbox_source``
     - ``False``
     - Register pre-instrumentation source in :mod:`linecache` so
       :func:`inspect.getsource` and tracebacks work for ``exec``/``eval``-compiled
       code.
   * - ``ast_rewriter_cls``
     - :class:`~pyccolo.AstRewriter`
     - The rewriter class to use; override to customize the transform.

.. seealso::

   The overridable **methods** (``should_instrument_file`` and friends) are on the
   :doc:`BaseTracer reference <tracer>`; :doc:`/concepts/composition` explains why
   transformational tracers set ``global_guards_enabled = False``.
