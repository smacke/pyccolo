Example gallery
===============

Each of the following ships under `pyccolo/examples/
<https://github.com/smacke/pyccolo/blob/master/pyccolo/examples>`_ as a
self-contained, tested tracer — great starting points to read and adapt. Where a
step-by-step **tutorial** builds one up from scratch, it's linked in the last
column.

.. list-table::
   :header-rows: 1
   :widths: 26 54 20

   * - Example
     - Demonstrates
     - Tutorial
   * - `coverage.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/coverage.py>`_
     - statement-level code coverage (``before_stmt`` +
       :meth:`~pyccolo.BaseTracer.should_instrument_file`)
     - :doc:`/tutorials/coverage_tracer`
   * - `optional_chaining.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/optional_chaining.py>`_
     - ``?.``, ``.?``, ``?.(``, and ``??`` optional chaining / nullish coalescing
       via ``AugmentationSpec``
     - :doc:`/tutorials/optional_chaining`
   * - `pipeline_tracer.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/pipeline_tracer.py>`_
     - ``|>`` / ``|>>`` pipeline operators (binop augmentation)
     - :doc:`/tutorials/pipeline_operator`
   * - `quick_lambda.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/quick_lambda.py>`_
     - MacroPy-style ``f[_ + _]`` quick lambdas
     - :doc:`/tutorials/quick_lambda`
   * - `lazy_imports.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/lazy_imports.py>`_
     - make (most) imports lazy, resolving on first use
     - :doc:`/tutorials/lazy_imports`
   * - `concolic.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/concolic.py>`_
     - concolic (concrete + symbolic) execution with a Z3 / brute-force solver
     - :doc:`/tutorials/concolic`
   * - `quasiquote.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/quasiquote.py>`_
     - MacroPy-style ``q[...]`` / ``u[...]`` quasiquotes
     -
   * - `block_lambda.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/block_lambda.py>`_,
       `func_block.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/func_block.py>`_,
       `brace_subscript.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/brace_subscript.py>`_
     - statement-bodied ``name{ ... }`` blocks (paired-delimiter augmentation)
     -
   * - `future_tracer.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/future_tracer.py>`_
     - implicit async: run assignments on a thread pool, unwrap futures on use
     -

The :doc:`/tutorials/watchpoint_debugger` tutorial builds a small
variable-watchpoint debugger that isn't in the gallery — a good from-scratch first
build.

In the wild
-----------

Pyccolo is the instrumentation engine behind several larger projects — good places
to see what it can do at scale:

- `ipyflow <https://github.com/ipyflow/ipyflow>`_ — a reactive Python kernel for
  Jupyter that tracks dataflow between cells using Pyccolo's dynamic analysis.
- `pipescript <https://github.com/smacke/pipescript>`_ — a pipe operator (``|>``),
  placeholder (``$``), and macro syntax for IPython/Jupyter, built entirely on
  Pyccolo's syntax augmentation and composable event handlers. It is the most
  complete showcase of syntax augmentation.
- `pycograd <https://github.com/smacke/pycograd>`_ — a small reverse-mode
  automatic-differentiation library that differentiates *ordinary* ``numpy`` code
  (no special "autodiff namespace"), using Pyccolo to trace the computation.
