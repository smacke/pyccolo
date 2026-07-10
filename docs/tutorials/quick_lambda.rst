Tutorial: quick lambdas (``f[_ + _]``)
======================================

MacroPy popularized a beautifully terse anonymous-function syntax: ``f[_ + _]`` is
a two-argument lambda, so ``f[_ + _](3, 4)`` is ``7``. In this tutorial you'll see
how Pyccolo implements it — a *subscript macro* that rewrites underscore
placeholders into a real lambda — and you'll drive the shipped
``QuickLambdaTracer`` to try every form. Syntax augmentation requires
Python ≥ 3.8.

The idea
--------

``f[...]`` looks like subscripting a name ``f``, but there is no ``f`` — the tracer
treats it as a macro. When the subscript's *slice* (the ``...`` inside the
brackets) is about to be evaluated, the tracer:

1. copies the slice expression's AST;
2. rewrites each bare ``_`` into a fresh parameter — ``_0``, ``_1``, ... — in order;
3. wraps the result in ``lambda _0, _1, ...: <slice>`` and evaluates it;
4. returns that lambda in place of the slice.

So ``f[_ + _]`` becomes ``lambda _0, _1: _0 + _1``, which you then call.

The key handler
---------------

The heart of it is a handler on ``before_subscript_slice`` — the event that fires
just before a subscript's index is evaluated — gated to fire only when the
subscripted name is one of the macro names (``f``, ``map``, ``filter``,
``reduce``, ...). Simplified, it looks like this:

.. code-block:: python

   import ast
   import pyccolo as pyc


   class QuickLambda(pyc.BaseTracer):
       macros = ("f",)

       @pyc.before_subscript_slice(
           when=lambda node: isinstance(node.value, ast.Name)
           and node.value.id in QuickLambda.macros,
           reentrant=True,
       )
       def build_lambda(self, _thunk, node, frame, *_, **__):
           body = ast.parse(...)             # a copy of the slice expression
           n = rewrite_underscores(body)     # `_` -> `_0`, `_1`, ...
           params = ", ".join(f"_{i}" for i in range(n))
           lam = compile_lambda(f"lambda {params}: <body>")
           return lambda: pyc.eval(lam, frame.f_globals, frame.f_locals)

The real implementation in ``pyccolo/examples/quick_lambda.py`` handles the fiddly
parts — nested quick lambdas, caching the compiled lambda, and the ``map`` /
``filter`` / ``reduce`` macros that additionally *apply* the lambda — but the
shape is exactly the above: recognize the macro, rewrite ``_`` placeholders, and
return the lambda. Note ``reentrant=True``: building the lambda evaluates
instrumented code, so the handler must be allowed to re-enter (see
:doc:`/guides/compose_tracers`).

Trying it out
-------------

Rather than rebuild all of that, let's drive the shipped tracer. Plain ``f[...]``
gives you an anonymous function; each ``_`` is the next positional argument:

.. testcode::

   import pyccolo as pyc
   from pyccolo.examples.quick_lambda import QuickLambdaTracer

   with QuickLambdaTracer:
       assert pyc.eval("f[_ + _](3, 4)") == 7      # lambda _0, _1: _0 + _1
       assert pyc.eval("f[_ * 2](10)") == 20       # lambda _0: _0 * 2

The ``map``, ``filter``, and ``reduce`` macros go one step further — they build the
lambda *and* apply it to an iterable:

.. testcode::

   with QuickLambdaTracer:
       assert pyc.eval("map[_ + 1]([1, 2, 3])") == [2, 3, 4]
       assert pyc.eval("filter[_ > 2]([1, 2, 3, 4])") == [3, 4]
       assert pyc.eval("reduce[_ + _]([1, 2, 3, 4])") == 10

Where to next
-------------

Quick lambdas shine inside pipelines: ``QuickLambdaTracer`` composes with the
shipped ``PipelineTracer``, so with both active
``(1, 2, 3) |> f[map(f[_ + 1], _)] |> list`` evaluates to ``[2, 3, 4]`` — layer
them with a nested ``with`` (see :doc:`/guides/compose_tracers`). Quick lambdas are
built on the same quasiquote machinery as the shipped ``quasiquote.py`` example, a
good next read for how Pyccolo manipulates ASTs as values.
