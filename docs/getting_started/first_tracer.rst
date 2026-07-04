Your first tracer
=================

Let's build the smallest interesting Pyccolo program and understand every line of
it. Here is a *tracer* that prints ``"Hello, world!"`` before every statement that
executes:

.. code-block:: python

   import pyccolo as pyc


   class HelloTracer(pyc.BaseTracer):
       @pyc.before_stmt
       def handle_stmt(self, *_, **__):
           print("Hello, world!")


   with HelloTracer:
       # prints "Hello, world!" 11 times
       pyc.exec("for _ in range(10): pass")

Three things are happening here.

The tracer class
----------------

Instrumentation is provided by a *tracer class* that inherits from
``pyc.BaseTracer``. The class rewrites Python source so that events of interest —
here, "a statement is about to execute" — trigger your code. You subscribe to an
event by decorating a method with it; ``@pyc.before_stmt`` registers
``handle_stmt`` as a :doc:`handler </reference/handlers>` for the ``before_stmt``
event. The ``*_, **__`` swallows the arguments this handler doesn't need (a
handler receives several — see :doc:`/reference/handlers`).

The class *is* the context manager
----------------------------------

Notice that we write ``with HelloTracer:`` — the **class itself**, not an
instance. You do not instantiate a tracer; ``BaseTracer`` is a singleton whose
class doubles as a context manager. Entering the ``with`` block activates the
tracer; leaving it deactivates it.

What is up with ``pyc.exec(...)``?
----------------------------------

A program's abstract syntax tree is fixed when the module is compiled. When our
script first started running, ``HelloTracer`` was not yet active, so any *unquoted*
Python in the same file was compiled **without** instrumentation — its statements
will never emit ``before_stmt``. To instrument code that lives in the same module
as the tracer definition, we hand it to ``pyc.exec(...)`` as a string (or AST), so
Pyccolo can rewrite and run it *while the tracer is active*.

``pyc.exec`` returns the resulting namespace as a dict, which is handy for
inspecting results. (Code in *other* modules can be instrumented at import time
instead — see :doc:`/howto/instrument_imports`. And ``sys``-level events like
``call`` don't need ``pyc.exec`` at all — see :doc:`/howto/sys_settrace`.) The
full story is in :doc:`/concepts/exec_and_scoping`.

Handlers can change behavior, not just observe it
-------------------------------------------------

A handler isn't limited to watching — for value-carrying events, the value it
returns *replaces* the value of the instrumented expression. Here is a tracer that
adds one to the result of every assignment's right-hand side:

.. code-block:: python

   import pyccolo as pyc


   class IncrementEveryAssignment(pyc.BaseTracer):
       @pyc.after_assign_rhs
       def handle(self, ret, *_, **__):
           return ret + 1


   with IncrementEveryAssignment:
       env = pyc.exec("x = 42")
       assert env["x"] == 43

The ``after_assign_rhs`` handler receives ``ret`` (the value the right-hand side
produced) and returns ``ret + 1``, which is what actually gets bound to ``x``.
Returning ``None`` (or nothing) would mean "don't override." This ability to
rewrite values in flight is what makes Pyccolo more than a profiler.

Where to go next
----------------

- :doc:`/concepts/model` — the events-and-handlers model, in depth.
- :doc:`/concepts/exec_and_scoping` — why ``pyc.exec`` is needed and how scoping
  works.
- :doc:`/tutorials/coverage_tracer` — build a real, statement-level code-coverage
  tool from scratch.
