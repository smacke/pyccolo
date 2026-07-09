Trace cells in IPython and Jupyter
==================================

**Goal:** instrument every notebook cell with a tracer, without wrapping code in
:func:`pyccolo.exec`.

Load the extension, then register a tracer:

.. code-block:: text

   In [1]: %load_ext pyccolo

   In [2]: import pyccolo as pyc
      ...: class HelloTracer(pyc.BaseTracer):
      ...:     @pyc.before_stmt
      ...:     def handle_stmt(self, *_, **__):
      ...:         print("Hello, world!")

   In [3]: %pyccolo register HelloTracer

   In [4]: for _ in range(3): pass
   Hello, world!
   Hello, world!
   Hello, world!
   Hello, world!

``%unload_ext pyccolo`` removes the instrumentation and restores the shell.

Registering a tracer
--------------------

``%pyccolo register`` and its Python equivalent,
:func:`pyccolo.register_ipython_tracer`, both accept four spellings:

.. code-block:: python

   pyc.register_ipython_tracer(HelloTracer)                    # a class
   pyc.register_ipython_tracer(HelloTracer.instance())         # an instance
   pyc.register_ipython_tracer("HelloTracer")                  # a name in the user namespace
   pyc.register_ipython_tracer("mypkg.tracers.HelloTracer")    # a fully qualified path

The extension must already be loaded; ``register_ipython_tracer`` raises
otherwise. The other subcommands are ``%pyccolo deregister <tracer>`` (or
``deregister all``) and ``%pyccolo list``.

Registration takes effect on the **next** cell: the current cell's code has
already been rewritten by the time your ``register`` call runs.

Ordering
--------

Registration order is canonical. The first tracer registered comes first in
``_TRACER_STACK``: its :doc:`syntax augmenter </howto/add_syntax>` rewrites the
source outermost, and its handlers see each event first.

.. code-block:: python

   pyc.register_ipython_tracer(A)
   pyc.register_ipython_tracer(B)
   # A's augmenter runs before B's; A's handlers run before B's

Composing with ipyflow
----------------------

`ipyflow <https://github.com/ipyflow/ipyflow>`_ drives pyccolo's cell tracing
driver itself rather than layering a second set of transformers on the shell, so
``%load_ext pyccolo`` and ``%load_ext ipyflow`` compose in either order, and a
cell is instrumented exactly once. ``%flow register_tracer`` and
``pyc.register_ipython_tracer`` write to the same registry, and ipyflow's own
tracers sort after yours. ``%unload_ext ipyflow`` hands the driver back, leaving
your tracers running under pyccolo alone.

How it works
------------

pyccolo installs one input transformer (which applies each tracer's syntax
augmenters and builds the cell's :class:`~pyccolo.AstRewriter`) and one AST
transformer (which runs that rewriter). It learns the cell's filename by wrapping
``shell.compile.cache`` — the caching compiler that mints it, called between the
two transforms — and adds it to each tracer's ``_tracing_enabled_files`` so the
:doc:`file filter </howto/instrument_imports>` admits the cell.

Tracing itself is *resident*: the tracers are entered once and merely
enabled/disabled around each cell, rather than pushed and popped every time.

Writing a tracer for the notebook
---------------------------------

Two things are worth knowing.

**Define tracers in a module, not a cell, when you can.** A tracer defined in a
cell has its handler's code object attributed to that cell's filename, so
co-resident tracers that use ``sys.settrace`` may follow execution into your
handler.

**A ``before_stmt`` handler alone would cost you ``Out[N]``.** Instrumenting a
statement wraps it in an ``if``, so the cell's body no longer *ends* in an
expression statement and IPython's ``last_expr`` interactivity suppresses the
output. pyccolo detects this and weaves in an ``after_module_stmt`` handler to
keep the trailing expression's value, so ``Out[N]`` behaves as usual. Nothing is
required of you.

JupyterLite
-----------

The extension imports only from ``IPython.core`` and never touches
``shell.kernel``, so it works unmodified under JupyterLite / Pyodide, where
``ipykernel`` is absent. Try it in the `in-browser demo
<https://smacke.github.io/pyccolo/lab/index.html?path=demo.ipynb>`_.
