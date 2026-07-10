Tutorial: make imports lazy
===========================

A big program can spend a surprising fraction of its startup time importing
modules it may never actually use on a given run. In this tutorial you'll follow
the design of the shipped ``lazy_imports.py`` tracer, which defers every
``import`` until the imported name is *first used* — turning eager import cost into
pay-per-use.

Because lazy imports only make sense for real modules loaded through the import
system (a name bound by ``import x as y`` becomes a fast local under
:func:`pyccolo.exec`, so this tracer is meant for import-time instrumentation or
the ``pyc`` CLI — see :doc:`/guides/tracing_real_programs`), the snippets here are
illustrative excerpts of ``pyccolo/examples/lazy_imports.py`` rather than runnable
doctests.

The plan
--------

1. When an ``import`` statement is about to run, **don't** run it. Instead bind a
   lightweight *placeholder* to each imported name and skip the statement.
2. When one of those names is first *loaded*, resolve it for real — import the
   module, cache it, and hand back the real object.
3. Use a :doc:`local guard </guides/guards_and_performance>` so that resolving
   costs one event per name, not one per use.

Step 1: a placeholder for an unresolved import
----------------------------------------------

The placeholder just remembers what it stands for and imports it on demand:

.. code-block:: python

   import importlib
   import pyccolo as pyc


   class _LazySymbol:
       def __init__(self, module_name):
           self.module_name = module_name
           self._resolved = None

       def unwrap(self):
           if self._resolved is None:
               with pyc.allow_reentrant_event_handling():
                   self._resolved = importlib.import_module(self.module_name)
           return self._resolved

``allow_reentrant_event_handling`` matters: the import runs instrumented code, and
without it the tracer would refuse to re-enter its own handlers.

Step 2: intercept the import statement
--------------------------------------

A ``before_stmt`` handler, statically gated to ``import`` statements, binds a
``_LazySymbol`` for each name and returns :data:`pyc.Pass` so the real import never
executes:

.. code-block:: python

   @pyc.before_stmt(
       when=pyc.Predicate(
           lambda node: isinstance(node, (ast.Import, ast.ImportFrom))
           and pyc.is_outer_stmt(node),
           static=True,
       )
   )
   def defer_import(self, _ret, node, frame, *_, **__):
       for alias in node.names:
           name = alias.asname or alias.name
           frame.f_globals[name] = _LazySymbol(alias.name)
       return pyc.Pass       # skip the actual import

The ``static=True`` predicate is resolved during the rewrite — whether a statement
is an ``import`` is decidable from syntax alone — so this handler adds nothing at
runtime except on the import lines themselves.

Step 3: resolve on first use
----------------------------

Now catch the first *load* of a deferred name. The ``guard=`` argument names a
per-variable guard so that, once resolved, later loads of the same name run
uninstrumented:

.. code-block:: python

   @pyc.init_module
   def init(self, _ret, node, frame, *_, **__):
       for guard in self.local_guards_by_module_id.get(id(node), []):
           frame.f_globals[guard] = False

   @pyc.load_name(guard=lambda node: f"{pyc.PYCCOLO_BUILTIN_PREFIX}_{node.id}")
   def resolve(self, ret, node, frame, _evt, guard, *_, **__):
       frame.f_globals[guard] = True          # never fire for this name again
       if isinstance(ret, _LazySymbol):
           real = ret.unwrap()
           frame.f_globals[node.id] = real     # replace the placeholder for good
           return real
       return ret

The first time ``math`` (say) is loaded, ``ret`` is the ``_LazySymbol``; we import
the module, rebind the global, activate the guard, and return the real module. The
guard means every subsequent ``math.*`` reference is a plain, uninstrumented global
lookup.

Putting it to work
------------------

Point the tracer at a module through the import machinery and its imports become
lazy:

.. code-block:: python

   from pyccolo.examples.lazy_imports import LazyImportTracer

   with LazyImportTracer.instance():
       import my_slow_starting_app     # its top-level imports are now deferred
       my_slow_starting_app.main()     # modules resolve as `main` touches them

Or run a whole script this way from the command line:

.. code-block:: bash

   pyc my_app.py -t pyccolo.examples.lazy_imports.ScriptLazyImportTracer

The full example handles ``from x import y``, relative imports, star imports
(which it declines to make lazy), attribute and subscript chains, and more — but
every path is a variation on the three steps above: defer, then resolve once on
first use.

Where to next
-------------

The local-guard technique that keeps resolution to one event per name is covered
in :doc:`/guides/guards_and_performance`; the import-time instrumentation that
makes this tracer apply to real modules is in
:doc:`/guides/tracing_real_programs`.
