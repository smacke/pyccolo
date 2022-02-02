Pyccolo
=======

[![CI Status](https://github.com/smacke/pyccolo/workflows/pyccolo/badge.svg)](https://github.com/smacke/pyccolo/actions)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/smacke/pyccolo/branch/master/graph/badge.svg?token=MGORH1IXLO)](https://codecov.io/gh/smacke/pyccolo)
[![License: BSD3](https://img.shields.io/badge/License-BSD3-maroon.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyccolo.svg)](https://pypi.org/project/pyccolo)
[![PyPI Version](https://img.shields.io/pypi/v/pyccolo.svg)](https://pypi.org/project/pyccolo)

Pyccolo is a library for declarative instrumentation in Python; i.e., it lets
you specify the *what* of the instrumentation you wish to perform, and takes
care of the *how* for you.  It aims to be *ergonomic*, *composable*, and
*portable*, by providing an intuitive interface, making it easy to layer
multiple levels of instrumentation, and allowing the same code to work across
multiple versions of Python (3.6 to 3.10), with few exceptions. Portability
across versions is accomplished by embedding instrumentation at the level of
source code (as opposed to bytecode-level instrumentation).

Pyccolo can be used (and has been used) to implement various kinds of dynamic analysis
tools and other instrumentation:
- Code coverage (see [pyccolo/examples/coverage.py](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/coverage.py))
- Syntactic macros such as quasiquotes (like [MacroPy's](https://macropy3.readthedocs.io/en/latest/reference.html#quasiquote)) or quick lambdas; see [pyccolo/examples/quasiquote.py](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/quasiquote.py) and [pyccolo/examples/quick_lambda.py](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/quick_lambda.py)
- Syntax-augmented Python (3.8 and up, see [pyccolo/examples/null_coalesce.py](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/null_coalesce.py))
- Dynamic dataflow analysis performed by [nbsafety](https://github.com/nbsafety-project/nbsafety)
- Tools to find unused imports at runtime (candidates for lazy importing)
- Tools to uncover [semantic memory leaks](http://ithare.com/java-vs-c-trading-ub-for-semantic-memory-leaks-same-problem-different-punishment-for-failure/)
- \<Your tool here!>

## Install

```bash
pip install pyccolo
```

## Hello World

Below is a simple script that uses Pyccolo to print "Hello, world!" before
every statement that executes:

```python
import pyccolo as pyc


class HelloTracer(pyc.BaseTracer):
    @pyc.before_stmt
    def handle_stmt(self, *_, **__):
        print("Hello, world!")


if __name__ == "__main__":
    with HelloTracer():
        # prints "Hello, world!" 11 times
        pyc.exec("for _ in range(10): pass")
```

Instrumentation is provided by a *tracer class* that inherit from
`pyccolo.BaseTracer`. This class rewrites Python source code with
instrumentation that triggers whenever events of interest occur, such as when a
statement is about to execute. By registering a handler with the associated
event (with the `@pyc.before_stmt` decorator, in this case), we can enrich our
programs with additional observability, or even alter their behavior
altogether.

### What is up with `pyc.exec(...)`?

A program's abstract syntax tree is fixed at import / compile time, and when
our script initially started running, the tracer was not active, so unquoted
Python in the same file will lack instrumentation. It is possible to instrument
modules at import time, but only when the imports are performed inside a
tracing context. Thus, we must quote any code appearing in the same module
where the tracer class was defined in order to instrument it.

## Composing tracers

A core feature of Pyccolo is that its instrumentation is *composable*.  It's
usually tricky to use two or more `ast.NodeTransformer` classes simultaneously
--- sometimes you can just have one inherit from the other, but if they both
define `visit` methods for the same AST node type, then typically you would
need to define a bespoke node transformer that uses logic from each base
transformer, handling corner cases to resolve incompatibilities.  With Pyccolo,
you simply compose the context managers of each tracer class whose
instrumentation you wish to use, and everything usually Just
Works<sup>TM</sup>:

```python
with tracer1:
    with tracer2:
        pyc.exec(...)
```

## Compatibility with sys.settrace(...)

Pyccolo is designed to support not only AST-level instrumentation, but also
instrumentation involving Python's [built in tracing
utilities](https://docs.python.org/3/library/sys.html#sys.settrace).
To use it, you simply register handlers for one of the corresponding
Pyccolo events (`call`, `line`, `return_`, `exception`, or `opcode`).
Here's a minimal example:

```python
import pyccolo as pyc


class SysTracer(pyc.BaseTracer):
    @pyc.call
    def handle_call(self, *_, **__):
        print("Pushing a stack frame!")

    @pyc.return_
    def handle_return(self, *_, **__):
        print("Popping a stack frame!")


if __name__ == "__main__":
    with SysTracer():
        def f():
            def g():
                return 42
            return g()
        # push, push, pop, pop
        answer_to_life_universe_everything = f()
```

Note that we didn't need to use `pyc.exec(...)` in the above example, because Python's built-in
tracing does not involve any AST-level transformations. If, however, we had registered handlers
for other events, such as `pyc.before_stmt`, we would need to use `pyc.exec(...)` to ensure those
handlers get called, when running code in the same file where our tracer class is defined.

### What if I'm already using sys.settrace(...) with my own tracing function?

Pyccolo is designed to be *composable*, and should execute both your tracing function as well
as any handlers defined in any active Pyccolo tracers. For example Pyccolo's unit tests for
`call` and `return` events work even when [coverage.py](https://coverage.readthedocs.io/)
is active (and without breaking it), which also uses Python's built-in tracing utilities.

## Instrumenting Imported Modules

Instrumentation is opt-in for modules imported within tracing contexts. To determine whether
a module gets instrumented, the method `should_instrument_file(...)` is called with the module's
corresponding filename as input. For example:

```python
class MyTracer(pyc.BaseTracer):
    def should_instrument_file(self, filename: str) -> bool:
        return filename.endswith("foo.py")
    
    # handlers, etc. defined below
    ...

with MyTracer():
    import foo  # contents of `foo` module get instrumented
    import bar  # contents of `bar` module do not get instrumented
```

Imports are instrumented by registering a custom finder / loader with `sys.meta_path`.
This loader ignores cached bytecode (which may possibly be uninstrumented), and avoids
generating *new* cached bytecode (which would be instrumented, possibly causing confusion
later when instrumentation is not desired).

## Command Line Interface

You can execute arbitrary scripts with instrumentation enabled with the `pyc` command line tool.
For example, to use the `NullCoalescer` tracer defined in [pyccolo/examples/null_coalesce.py](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/null_coalesce.py),
you can call `pyc` as follows, given some example script `bar.py`:

```python
# bar.py
bar = None
# prints `None` since bar?.foo coalesces to `None`
print(bar?.foo)
```

```bash
> pyc bar.py -t pyccolo.examples.NullCoalescer
```

You can also run `bar` as a module (indeed, `pyc` performs this internally when provided a file):

```bash
> pyc -m bar -t pyccolo.examples.NullCoalescer
```

Note that you can specify multiple tracer classes after the `-t` argument;
in case you were not already aware, Pyccolo is composable! :)

The above example demonstrates a tracer class that performs syntax augmentation on its
instrumented Python source to modify the default Python syntax. This feature is available
only on Python >= 3.8 for now and is lacking documentation for the moment, but you can
see some examples in the [test_syntax_augmentation.py](https://github.com/smacke/pyccolo/blob/master/test/test_syntax_augmentation.py) unit tests.

## More Events

Pyccolo handlers can be registered for many kinds of events. Some of the more common ones are:
- `pyc.before_stmt`, emitted before a statement executes;
- `pyc.after_stmt`, emitted after a statement executes;
- `pyc.before_attribute_load`, emitted in [load contexts](https://docs.python.org/3/library/ast.html#ast.Load) before an attribute is accessed;
- `pyc.after_attribute_load`, emitted in load contexts after an attribute is accessed;
- `pyc.load_name`, emitted when a variable is used in a load context (e.g. `foo` in `bar = foo.baz`);
- `pyc.call` and `pyc.return_`, two non-AST trace events built-in to Python.

There are many different Pyccolo events, and more are always being added. See
[pyccolo/trace_events.py](https://github.com/smacke/pyccolo/blob/master/pyccolo/trace_events.py)
for a full list.

Note that, for AST events, Python source is only transformed to emit some event
when there is at least one tracer active that has at least one handler
registered for that event. This prevents the transformed source from becoming
extremely bloated when only a few events are needed.

## Handler Interface

Every Pyccolo handler is passed four positional arguments:
1. The return value, for instrumented expressions;
2. The AST node (or node id, if using `register_raw_handler(...)`, or `None`, for `sys` events);
3. The stack frame, at the point where instrumentation kicks in;
4. The event (useful when the same handler is registered for multiple events).

Some events pass additional keyword arguments, which I'm still in the process
of documenting, but the above four tend to suffice for most use cases.

Not every handler receives a return value; for example, this argument is always
`None` for `pyc.after_stmt` handlers. For certain handlers, the return value
can be overridden. For example, by returning a value in a
`pyc.before_attribute_load`, we override the object whose attribute is
accessed. If we return nothing or `None`, then we do not override this object.
(If we actually want to override it as `None` for some reason, then we can
return `pyc.Null`.) For a particular event, handler return values compose with
other handlers defined on the same tracer class as well as with handlers
defined on other tracer classes.

## Performance

Pyccolo instrumentation adds significant overhead to Python. In some
cases, this overhead can be partially mitigated if, for example, you only need
instrumentation the first time a statement runs. In such cases, you can
deactivate instrumentation after, e.g., the first time a function executes, or
after the first iteration in a loop for that respective function or loop, so
that further calls (iterations, respectively) use uninstrumented code with all
the mighty performance of native Python. This is implemented by activating
"guards" associated with the function or loop, as in the below example:

```python
class TracesOnce(pyc.BaseTracer):
    @pyc.register_raw_handler((pyc.after_for_loop_iter, pyc.after_while_loop_iter))
    def after_loop_iter(self, *_, guard, **__):
        self.activate_guard(guard)

    @pyc.register_raw_handler(pyc.after_function_execution)
    def after_function_exec(self, *_, guard, **__):
        self.activate_guard(guard)
```

Subsequent calls / iterations will be instrumented only after calling
`self.deactivate_guard(...)` on the associated function / loop guard.

## License
Code in this project licensed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
