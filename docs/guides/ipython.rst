IPython and Jupyter
===================

In a notebook you don't want to wrap every cell in :func:`pyccolo.exec`. Pyccolo
ships an IPython extension that instruments each cell automatically: load it, then
register one or more tracers, and every subsequent cell is traced as if it had
been run through ``pyc.exec`` — including notebook niceties like ``Out[N]`` and
composition with other cell-level tools such as `ipyflow
<https://github.com/ipyflow/ipyflow>`_.

Load the extension and register a tracer
----------------------------------------

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

``%unload_ext pyccolo`` removes the instrumentation and restores the shell to
exactly how it was.

The four ways to name a tracer
------------------------------

``%pyccolo register`` and its programmatic twin
:func:`pyccolo.register_ipython_tracer` accept the tracer in four spellings, so
you can register from a cell, a variable, or a library:

.. code-block:: python

   pyc.register_ipython_tracer(HelloTracer)                  # a class
   pyc.register_ipython_tracer(HelloTracer.instance())       # an instance
   pyc.register_ipython_tracer("HelloTracer")                # a name in the user namespace
   pyc.register_ipython_tracer("mypkg.tracers.HelloTracer")  # a dotted path

The extension must already be loaded or ``register_ipython_tracer`` raises. The
companion subcommands are ``%pyccolo deregister <tracer>`` (or ``deregister
all``) and ``%pyccolo list``, plus :func:`pyccolo.deregister_ipython_tracer` and
:func:`pyccolo.registered_ipython_tracers`.

Registration takes effect on the **next** cell: by the time your ``register`` call
runs, the current cell has already been rewritten.

Registration order is canonical
-------------------------------

The first tracer registered comes first in the tracer stack: its :doc:`syntax
augmenter </guides/add_syntax>` rewrites the source outermost, and its handlers
see each event first.

.. code-block:: python

   pyc.register_ipython_tracer(A)
   pyc.register_ipython_tracer(B)
   # A's augmenter runs before B's; A's handlers run before B's

Composing with ipyflow
----------------------

ipyflow drives Pyccolo's cell-tracing driver directly instead of layering a second
set of transformers on the shell, so ``%load_ext pyccolo`` and ``%load_ext
ipyflow`` compose in either order and a cell is instrumented exactly once.
``%flow register_tracer`` and ``pyc.register_ipython_tracer`` write to the same
registry, and ipyflow's own tracers sort after yours. ``%unload_ext ipyflow``
hands the driver back, leaving your tracers running under Pyccolo alone. The
handoff primitives — :func:`pyccolo.take_over_ipython_driver` and
:func:`pyccolo.release_ipython_driver` — are what make this seamless.

How it works
------------

Pyccolo installs one input transformer (which applies each tracer's syntax
augmenters and builds the cell's :class:`~pyccolo.AstRewriter`) and one AST
transformer (which runs that rewriter). It learns the cell's filename by wrapping
``shell.compile.cache`` — the caching compiler that mints it, called between the
two transforms — and adds it to each tracer's tracing-enabled files so the
:doc:`file filter </guides/tracing_real_programs>` admits the cell. Tracing is
*resident*: tracers are entered once and merely enabled/disabled around each cell,
rather than pushed and popped every time.

Two things worth knowing when you write a tracer for the notebook
-----------------------------------------------------------------

**Define tracers in a module, not a cell, when you can.** A tracer defined in a
cell has its handler's code object attributed to that cell's filename, so
co-resident tracers that use ``sys.settrace`` may follow execution into your
handler.

**You don't lose** ``Out[N]``. Instrumenting a statement wraps it in an ``if``, so
a cell body no longer *ends* in a bare expression statement — which would normally
make IPython's ``last_expr`` interactivity suppress the output. Pyccolo detects
this and weaves in an ``after_module_stmt`` handler that preserves the trailing
expression's value, so ``Out[N]`` behaves as usual. Nothing is required of you.

JupyterLite
-----------

The extension imports only from ``IPython.core`` and never touches
``shell.kernel``, so it runs unmodified under JupyterLite / Pyodide, where
``ipykernel`` is absent. The `in-browser demo
<https://smacke.github.io/pyccolo/lab/index.html?path=demo.ipynb>`_ is exactly
this path — Pyccolo tracing cells, entirely client-side.
