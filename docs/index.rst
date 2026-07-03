Welcome to Pyccolo's documentation!
===================================

**Pyccolo** (pronounced like the instrument "piccolo") is a library for
*declarative instrumentation* in Python: you specify the **what** of the
instrumentation you wish to perform, and Pyccolo takes care of the **how**. It
brings metaprogramming to everybody through general, event-emitting AST
transformations, and aims to be:

- **ergonomic** — you subclass :class:`pyccolo.BaseTracer` and decorate a
  handler; there's no bytecode to patch and no :class:`ast.NodeTransformer` to
  hand-write;
- **composable** — layering multiple, independently-written instrumentations
  usually Just Works (see :doc:`composing_tracers`);
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

**New here?** Start with :doc:`installation`, then walk through
:doc:`quickstart`. From there, :doc:`events_and_handlers` is the conceptual
heart of the library. Looking for a specific class or function? Jump to the
:doc:`api_reference`, or the :doc:`cli` for the ``pyc`` command-line tool.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   events_and_handlers
   composing_tracers
   predicates
   syntax_augmentation
   source_to_source
   sys_settrace
   imported_modules
   performance

.. toctree::
   :maxdepth: 2
   :caption: Reference

   cli
   examples
   api_reference
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
