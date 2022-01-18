Pyccolo
=======

[![CI Status](https://github.com/smacke/pyccolo/workflows/pyccolo/badge.svg)](https://github.com/smacke/pyccolo/actions)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/smacke/pyccolo/branch/master/graph/badge.svg?token=MGORH1IXLO)](https://codecov.io/gh/smacke/pyccolo)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: BSD3](https://img.shields.io/badge/License-BSD3-maroon.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyccolo.svg)](https://pypi.org/project/pyccolo)
[![PyPI Version](https://img.shields.io/pypi/v/pyccolo.svg)](https://pypi.org/project/pyccolo)

Pyccolo is a software library for embedding source-level instrumentation in Python
(as opposed to bytecode-level). It aims to be *ergonomic*, *composable*, and *portable*, by providing an intuitive
interface, making it easy to layer multiple levels of instrumentation, and allowing the same code to work across
multiple versions of Python (3.6 to 3.10), with few exceptions.

## Hello World

Below is a simple script that uses Pyccolo to count the number of statements that execute in a thunk:

```python
import pyccolo as pyc


class TrackExecutedStatements(pyc.BaseTracer):
    seen_stmts = set()

    @pyc.before_stmt
    def handle_stmt(self, _skip, ast_node, *_, **__):
        # handler positional arguments are:
        # 1. the return value, if the instrumented node is an expression (N/A here)
        # 2. the ast node
        # 3. stack frame where the handler was triggered (unused here)
        # 4. the event we're handling (in case the same handler is used
        #    for multiple events and we need to distinguish them; also
        #    unused here).
        self.seen_stmts.add(id(ast_node))


if __name__ == "__main__":
    tracer = TrackExecutedStatements()
    with tracer:
        z = pyc.exec(
            """
            x = 42
            y = x + 1
            if x + y < 100:
                z = x + y
            else:
                z = 99
            # prints 5 -- includes every ast statement except `z = 99`,
            #  and includes itself
            print("num stmts executed: %d" % len(tracer.seen_stmts))
            """
        )["z"]
    print("z = %d" % z)
```

This program is enough to demonstrate the core concepts and features of Pyccolo. Instrumentation is provided by a *
tracer class* that inherit from `pyccolo.BaseTracer`. This class rewrites Python source code with additional triggers
that execute when events of interest kick off, such as when a statement is about to execute, as in the above example. By
registering a handler with the associated event (with the `@pyc.before_stmt` decorator, in this case), we can enrich
our programs with additional observability, or even alter their behavior altogether.

## What is up with the string in `pyc.exec(...)`?

The instrumentation is activated by using an instance of the tracer class as a context manager. Note that we had to wrap
the instrumented code in a string. This is because a program's abstract syntax tree is fixed at import / compile time,
and when our script initially started running, the tracer was not active, so all Python code that is not quoted will
lack the tracing instrumentation. It is possible for Pyccolo tracers to opt in to tracing for modules imported within
their context managers to have instrumentation added, but to use the instrumentation in the same script where the tracer
class was defined, we typically need to quote the instrumented portion.

## License
Code in this project licensed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
