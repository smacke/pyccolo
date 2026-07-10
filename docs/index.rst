Pyccolo
=======

**Pyccolo** (pronounced like the instrument "piccolo") lets you change what
Python *does* — observe it, rewrite values as they flow, or teach the language
brand-new syntax — without touching bytecode, patching the interpreter, or
hand-writing an :class:`ast.NodeTransformer`. You write a small class, decorate a
method with the moment you care about, and Pyccolo rewrites the program's source
so that moment calls your code.

That single idea scales a long way. The same twenty-line building block underlies
statement-level code coverage, a reactive Jupyter kernel, a pipe operator, lazy
imports, optional chaining, and reverse-mode autodiff over ordinary NumPy — all
listed under :doc:`examples </examples>`.

A tracer in fifteen seconds
---------------------------

Here is a tracer that makes every ``float`` literal *exact* by promoting it to a
:class:`~decimal.Decimal` — a real behavioral change in one handler:

.. testcode::

   from decimal import Decimal
   import pyccolo as pyc


   class ExactFloats(pyc.BaseTracer):
       @pyc.after_float
       def to_decimal(self, ret, *_, **__):
           return Decimal(str(ret))


   with ExactFloats:
       print(pyc.exec("x = 0.1 + 0.2")["x"])

.. testoutput::

   0.3

``@pyc.after_float`` subscribes ``to_decimal`` to the *"a float literal was just
evaluated"* event; whatever the handler returns **replaces** that value. No float
in the ``pyc.exec`` block escaped the substitution, and nothing outside the
``with`` block was affected. That is the whole model — :doc:`events and handlers
</concepts/model>` — and everything else is built on it.

Why Pyccolo
-----------

- **Ergonomic** — subscribe to an event by decorating a method. There is no
  bytecode to patch and no AST visitor to write by hand.
- **Composable** — independently-written tracers layer just by nesting their
  ``with`` blocks; their return values thread through in order. See
  :doc:`/guides/compose_tracers`.
- **Portable** — instrumentation lives at the level of *source code*, so the same
  tracer runs on CPython 3.6 through 3.14 (a few features need 3.8+).
- **Pay-as-you-go** — Pyccolo only rewrites a construct when some active tracer
  actually listens for it, so a hundred available events cost nothing until you
  use one.

Try it in your browser
----------------------

You don't have to install anything to get a feel for it: the `live JupyterLite
demo <https://smacke.github.io/pyccolo/lab/index.html?path=demo.ipynb>`_ runs
Pyccolo entirely in your browser.

How this documentation is organized
-----------------------------------

- **Start here** installs Pyccolo and walks through your first tracer line by
  line.
- **Guides** are task-oriented and example-dense: overriding values, composing
  tracers, guards and performance, running under imports/functions/the CLI,
  source-to-source rewriting, adding syntax, and the IPython integration.
- **Tutorials** build complete tools end to end — a coverage tracer, optional
  chaining, a pipe operator, quick lambdas, lazy imports, a concolic-execution
  engine, and a variable watchpoint debugger.
- **How it works** is the mental model: the rewrite pipeline, why tracers
  compose, and how ``pyc.exec`` and scoping fit together.
- **Reference** is the precise, code-derived catalog of every event, method, and
  option.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   getting_started/installation
   getting_started/first_tracer

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/observe_and_override
   guides/compose_tracers
   guides/guards_and_performance
   guides/tracing_real_programs
   guides/source_to_source
   guides/add_syntax
   guides/ipython

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/coverage_tracer
   tutorials/optional_chaining
   tutorials/pipeline_operator
   tutorials/quick_lambda
   tutorials/lazy_imports
   tutorials/concolic
   tutorials/watchpoint_debugger

.. toctree::
   :maxdepth: 2
   :caption: How it works

   concepts/model
   concepts/composition
   concepts/exec_and_scoping

.. toctree::
   :maxdepth: 2
   :caption: Reference

   reference/events
   reference/handlers
   reference/registration
   reference/tracer
   reference/config
   reference/predicates
   reference/augmentation
   reference/source_api
   reference/cli
   reference/utilities

.. toctree::
   :maxdepth: 1
   :caption: More

   examples
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
