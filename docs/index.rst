Pyccolo
=======

**Pyccolo** (pronounced like the instrument "piccolo") is a library for
*declarative instrumentation* in Python: you specify the **what** of the
instrumentation you wish to perform, and Pyccolo takes care of the **how**. It
brings metaprogramming to everybody through general, event-emitting AST
transformations, and aims to be:

- **ergonomic** — you subclass :class:`pyccolo.BaseTracer` and decorate a
  handler; there's no bytecode to patch and no :class:`ast.NodeTransformer` to
  hand-write;
- **composable** — layering multiple, independently-written instrumentations
  usually Just Works (see :doc:`/concepts/composition`);
- **portable** — the same code runs across Python 3.6 through 3.14, with few
  exceptions, because instrumentation is embedded at the level of *source code*
  rather than bytecode.

Here is the smallest interesting program: a tracer that prints ``"Hello,
world!"`` before every statement that executes.

.. code-block:: python

   import pyccolo as pyc


   class HelloTracer(pyc.BaseTracer):
       @pyc.before_stmt
       def handle_stmt(self, *_, **__):
           print("Hello, world!")


   with HelloTracer:
       # prints "Hello, world!" 11 times
       pyc.exec("for _ in range(10): pass")

Where to go from here
---------------------

This documentation is organized in four tracks, following the `Diátaxis
<https://diataxis.fr/>`_ framework:

- **Getting started** — install Pyccolo and build your first tracer.
- **Tutorials** — learning-oriented, end-to-end builds you follow start to finish.
- **How-to guides** — short, focused recipes for a specific task.
- **Concepts** — the mental model: how the instrumentation actually works.
- **Reference** — the precise, code-derived catalog of every event, method, and
  option.

If you are brand new, read :doc:`/getting_started/installation` then
:doc:`/getting_started/first_tracer`. If you want to understand *why* it works,
start with :doc:`/concepts/model`. If you are looking something up, jump to the
:doc:`/reference/events` or :doc:`/reference/tracer`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   getting_started/installation
   getting_started/first_tracer

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/coverage_tracer
   tutorials/optional_chaining

.. toctree::
   :maxdepth: 2
   :caption: How-to guides

   howto/override_values
   howto/compose_tracers
   howto/conditional_handlers
   howto/instrument_imports
   howto/add_syntax
   howto/source_to_source
   howto/sys_settrace
   howto/performance_guards

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/model
   concepts/rewrite_pipeline
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
