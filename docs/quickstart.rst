Quickstart
==========

This page walks through the "Hello, world!" tracer from the front page in
detail, and explains the one piece of Pyccolo that surprises newcomers: why the
traced code is wrapped in :func:`pyccolo.exec`.

Hello, world!
-------------

.. code-block:: python

   import pyccolo as pyc


   class HelloTracer(pyc.BaseTracer):
       @pyc.before_stmt
       def handle_stmt(self, *_, **__):
           print("Hello, world!")


   if __name__ == "__main__":
       with HelloTracer:
           # prints "Hello, world!" 11 times
           pyc.exec("for _ in range(10): pass")

Instrumentation is provided by a *tracer class* that inherits from
:class:`pyccolo.BaseTracer`. Subclassing it rewrites Python source code with
instrumentation that fires whenever events of interest occur — here, whenever a
statement is about to execute. By registering a handler for the associated event
(with the ``@pyc.before_stmt`` decorator), we enrich the program with additional
observability, or even alter its behavior altogether.

A tracer class is used as a **context manager**. Inside the ``with HelloTracer:``
block, instrumentation is active; outside it, ordinary Python runs unchanged.
(The class *itself* is the context manager — you do not instantiate it.)

Why ``pyc.exec(...)``?
----------------------

A program's abstract syntax tree is fixed at import / compile time. When our
script initially started running, the tracer was not active, so unquoted Python
in the *same file* will lack instrumentation — the eleven "Hello, world!" lines
come from the code passed to :func:`pyccolo.exec`, not from the ``for`` loop
being written directly in the module body.

It *is* possible to instrument modules at import time (see
:doc:`imported_modules`), but only when the imports are performed inside a
tracing context. Thus, to instrument code appearing in the same module where the
tracer class is defined, we must "quote" it — pass it as a string (or a
pre-parsed AST) to :func:`pyccolo.exec`, :func:`pyccolo.eval`, or
:func:`pyccolo.execute`.

:func:`pyccolo.exec` returns the resulting namespace as a dictionary, so you can
inspect what the traced code produced:

.. code-block:: python

   with HelloTracer:
       env = pyc.exec("x = 1\ny = x + 1")

   assert env["y"] == 2

Note that ``sys``-level events (``call``, ``line``, ``return_``, ``exception``,
``opcode``) do **not** require ``pyc.exec``, because they do not involve any
AST-level transformation — see :doc:`sys_settrace`.

Where to go next
----------------

- :doc:`events_and_handlers` — the event taxonomy, the handler signature, and
  how a handler can *override* (not just observe) a value.
- :doc:`composing_tracers` — layering multiple tracers.
- :doc:`examples` — a gallery of complete, runnable tracers to adapt.
