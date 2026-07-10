Tutorial: build a coverage tracer
=================================

In this tutorial you'll build a working **statement-level code-coverage** tool
from scratch. Along the way you'll meet raw handlers, file filtering, sandbox
detection, and the tracing lifecycle hooks — the same ingredients that power the
``coverage.py`` example shipped with Pyccolo.

The idea is simple: every time a statement is *about* to run, record it. At the
end, compare the set of statements we saw against the total number of statements
in each file.

Step 1: record every statement
------------------------------

Start with a :class:`pyccolo.BaseTracer` subclass that handles ``before_stmt``.
We only need to know *which* statement fired — its identity — so we register a
**raw** handler with ``pyc.register_raw_handler``. A raw handler receives the
integer node id instead of a resolved :class:`ast.AST` node, which is exactly what
we want to use as a dictionary/set key (and it's the fastest path). See
:doc:`/reference/registration` for the difference.

.. code-block:: python

   from collections import Counter

   import pyccolo as pyc


   class CoverageTracer(pyc.BaseTracer):
       bytecode_caching_allowed = False

       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.seen_stmts = set()
           self.stmt_count_by_fname = Counter()

       @pyc.register_raw_handler(pyc.before_stmt)
       def handle_stmt(self, _ret, stmt_id, frame, *_, **__):
           fname = frame.f_code.co_filename
           if stmt_id not in self.seen_stmts:
               self.stmt_count_by_fname[fname] += 1
               self.seen_stmts.add(stmt_id)

Why ``bytecode_caching_allowed = False``? Pyccolo can cache the instrumented
bytecode it compiles. For most tracers that's a welcome speedup, but a coverage
tool must see *every* statement on *every* run — a stale or partially-cached
module could omit statements from the count. Turning caching off keeps the
picture honest. (See :doc:`/reference/config` for the full list of these
class-level switches.)

Step 2: choose which files to cover
-----------------------------------

By default, imported modules are not instrumented. You opt files in by overriding
:meth:`~pyccolo.BaseTracer.should_instrument_file`, which is called with each
candidate module's filename. Return ``True`` for the files you want to measure:

.. code-block:: python

   def should_instrument_file(self, filename: str) -> bool:
       if "test/" in filename or "examples" in filename:
           return False
       return "pyccolo" in filename

Here we cover the ``pyccolo`` package itself while skipping tests and examples.
See :doc:`/guides/tracing_real_programs` for the mechanics of import instrumentation.

Step 3: ignore sandbox frames
-----------------------------

Code you run through ``pyc.exec`` / ``pyc.eval`` is compiled under a synthetic
"sandbox" filename. If you only care about coverage of real files on disk, filter
those frames out by testing the frame's filename against
``pyc.SANDBOX_FNAME_PREFIX``:

.. code-block:: python

   @pyc.register_raw_handler(pyc.before_stmt)
   def handle_stmt(self, _ret, stmt_id, frame, *_, **__):
       fname = frame.f_code.co_filename
       if fname.startswith(pyc.SANDBOX_FNAME_PREFIX):
           return
       if stmt_id not in self.seen_stmts:
           self.stmt_count_by_fname[fname] += 1
           self.seen_stmts.add(stmt_id)

Step 4: report at the end
-------------------------

To emit a report when tracing finishes, override
:meth:`~pyccolo.BaseTracer.exit_tracing_hook`. We need the *total* number of
statements per file to compute a ratio; a small :class:`ast.NodeVisitor` that
counts :class:`ast.stmt` nodes does the job:

.. code-block:: python

   import ast


   class CountStatementsVisitor(ast.NodeVisitor):
       def __init__(self):
           self.num_stmts = 0

       def generic_visit(self, node):
           if isinstance(node, ast.stmt) and not isinstance(node, ast.Raise):
               self.num_stmts += 1
           super().generic_visit(node)


   def count_statements(path: str) -> int:
       with open(path) as f:
           visitor = CountStatementsVisitor()
           visitor.visit(ast.parse(f.read()))
       return visitor.num_stmts

.. code-block:: python

   def exit_tracing_hook(self) -> None:
       total = 0
       for fname in sorted(self.stmt_count_by_fname):
           if fname.startswith(pyc.SANDBOX_FNAME_PREFIX):
               continue
           seen = self.stmt_count_by_fname[fname]
           in_file = count_statements(fname)
           total += in_file
           print(f"{fname}: seen={seen}, total={in_file}, ratio={seen / in_file:.3f}")
       if total:
           print(f"overall ratio: {len(self.seen_stmts) / total:.3f}")

Step 5: run it
--------------

Activate the tracer as a context manager and run some code under it. Coverage of
imported modules works because the tracer is active while the import happens:

.. code-block:: python

   with CoverageTracer.instance():
       import some_package  # its statements are now counted as they execute

To measure coverage of a whole test suite, drive the test runner from *inside* the
tracing context and install the import hook so freshly-imported modules get
instrumented. The shipped example does exactly this with pytest:

.. code-block:: python

   from pyccolo.import_hooks import patch_meta_path

   with CoverageTracer.instance():
       with patch_meta_path(pyc._TRACER_STACK):
           pytest.console_main()

That's a complete, if minimal, coverage tool — a couple of handlers and two
lifecycle hooks. The full version lives in ``pyccolo/examples/coverage.py``.

Where to next
-------------

- Coverage adds real overhead. :doc:`/guides/guards_and_performance` shows how to
  deactivate instrumentation once you've seen a statement for the first time.
- The :doc:`/reference/tracer` page catalogs every hook and method you can
  override.
