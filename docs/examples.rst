Example gallery
===============

Each of the following ships under
`pyccolo/examples/ <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples>`_
as a self-contained, tested tracer — great starting points to adapt.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Example
     - Demonstrates
   * - `coverage.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/coverage.py>`_
     - statement-level code coverage (``before_stmt``, ``should_instrument_file``)
   * - `optional_chaining.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/optional_chaining.py>`_
     - ``?.``, ``.?``, ``??`` optional chaining / nullish coalescing via
       :class:`~pyccolo.AugmentationSpec`
   * - `pipeline_tracer.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/pipeline_tracer.py>`_
     - ``|>`` / ``|>>`` pipeline operators (binop augmentation)
   * - `quick_lambda.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/quick_lambda.py>`_
     - MacroPy-style ``f[_ + _]`` quick lambdas
   * - `quasiquote.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/quasiquote.py>`_
     - MacroPy-style ``q[...]`` / ``u[...]`` quasiquotes
   * - `block_lambda.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/block_lambda.py>`_,
       `func_block.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/func_block.py>`_,
       `brace_subscript.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/brace_subscript.py>`_
     - statement-bodied ``name{ ... }`` blocks (paired-delimiter augmentation)
   * - `lazy_imports.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/lazy_imports.py>`_
     - make (most) imports lazy, resolving on first use
   * - `future_tracer.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/future_tracer.py>`_
     - implicit async: run assignments on a thread pool, unwrap futures on use
   * - `concolic.py <https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/concolic.py>`_
     - concolic (concrete + symbolic) execution with a Z3 / brute-force solver

In the wild
-----------

Beyond the bundled examples, Pyccolo is the instrumentation engine behind
several larger projects — good places to see what it can do at scale:

- `ipyflow <https://github.com/ipyflow/ipyflow>`_ — a reactive Python kernel for
  Jupyter that tracks dataflow between cells using Pyccolo's dynamic analysis.
- `pipescript <https://github.com/smacke/pipescript>`_ — a pipe operator
  (``|>``), placeholder (``$``), and macro syntax for IPython/Jupyter, built
  entirely on Pyccolo's syntax augmentation and composable event handlers.
- `pycograd <https://github.com/smacke/pycograd>`_ — a small reverse-mode
  automatic-differentiation library that differentiates *ordinary* ``numpy`` code
  (no special "autodiff namespace"), using Pyccolo to trace the computation.

People have also built statement-level coverage, syntactic macros (quasiquotes,
quick lambdas), syntax-augmented Python (optional chaining, pipeline operators),
lazy imports, concolic execution, and tools to uncover semantic memory leaks.
