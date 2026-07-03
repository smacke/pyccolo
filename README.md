Pyccolo
=======

[![CI Status](https://github.com/smacke/pyccolo/workflows/pyccolo/badge.svg)](https://github.com/smacke/pyccolo/actions)
[![Checked with mypy](http://www.mypy-lang.org/static/mypy_badge.svg)](http://mypy-lang.org/)
[![codecov](https://codecov.io/gh/smacke/pyccolo/branch/master/graph/badge.svg?token=MGORH1IXLO)](https://codecov.io/gh/smacke/pyccolo)
[![License: BSD3](https://img.shields.io/badge/License-BSD3-maroon.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Python Versions](https://img.shields.io/pypi/pyversions/pyccolo.svg)](https://pypi.org/project/pyccolo)
[![PyPI Version](https://img.shields.io/pypi/v/pyccolo.svg)](https://pypi.org/project/pyccolo)

Pyccolo (pronounced like the instrument "piccolo") is a library for declarative
instrumentation in Python; i.e., it lets you specify the *what* of the
instrumentation you wish to perform, and takes care of the *how* for you. It
brings metaprogramming to everybody via general, event-emitting AST
transformations, and aims to be:

- ***ergonomic*** — you subclass `pyc.BaseTracer` and decorate a handler; there's
  no bytecode to patch and no `ast.NodeTransformer` to hand-write;
- ***composable*** — layering multiple, independently-written instrumentations
  usually Just Works<sup>TM</sup> (more on this [below](#composing-tracers));
- ***portable*** — the same code runs across Python 3.6 through 3.14, with few
  exceptions, because instrumentation is embedded at the level of *source code*
  rather than bytecode.

## In the wild

Pyccolo is the instrumentation engine behind several projects — good places to
see what it can do at scale:

- [**ipyflow**](https://github.com/ipyflow/ipyflow) — a reactive Python kernel
  for Jupyter that tracks dataflow between cells using Pyccolo's dynamic
  analysis.
- [**pipescript**](https://github.com/smacke/pipescript) — a pipe operator
  (`|>`), placeholder (`$`), and macro syntax for IPython/Jupyter, built entirely
  on Pyccolo's syntax-augmentation and composable event handlers.
- [**pycograd**](https://github.com/smacke/pycograd) — a small reverse-mode
  automatic-differentiation library that differentiates *ordinary* `numpy` code
  (no special "autodiff namespace"), using Pyccolo to trace the computation.
- \<Your tool here!>

Other things people have built with Pyccolo include statement-level code
coverage, syntactic macros (quasiquotes, quick lambdas), syntax-augmented Python
(optional chaining, pipeline operators), lazy imports, concolic execution, and
tools to uncover [semantic memory
leaks](http://ithare.com/java-vs-c-trading-ub-for-semantic-memory-leaks-same-problem-different-punishment-for-failure/).
See the [example gallery](#example-gallery) for runnable versions of many of
these.

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
    with HelloTracer:
        # prints "Hello, world!" 11 times
        pyc.exec("for _ in range(10): pass")
```

Instrumentation is provided by a *tracer class* that inherits from
`pyccolo.BaseTracer`. This class rewrites Python source code with
instrumentation that triggers whenever events of interest occur, such as when a
statement is about to execute. By registering a handler with the associated
event (with the `@pyc.before_stmt` decorator, in this case), we can enrich our
programs with additional observability, or even alter their behavior altogether.

### What is up with `pyc.exec(...)`?

A program's abstract syntax tree is fixed at import / compile time, and when our
script initially started running, the tracer was not active, so unquoted Python
in the same file will lack instrumentation. It is possible to instrument modules
at import time (see [below](#instrumenting-imported-modules)), but only when the
imports are performed inside a tracing context. Thus, we must quote any code
appearing in the same module where the tracer class was defined in order to
instrument it.

## The model: events and handlers

Pyccolo exposes a fine-grained taxonomy of over 100 events (see
[pyccolo/trace_events.py](https://github.com/smacke/pyccolo/blob/master/pyccolo/trace_events.py)
for the full list). Some of the more common ones:

- `pyc.before_stmt` / `pyc.after_stmt`, emitted around statements;
- `pyc.before_attribute_load` / `pyc.after_attribute_load`, emitted in
  [load contexts](https://docs.python.org/3/library/ast.html#ast.Load) around
  attribute accesses;
- `pyc.load_name`, emitted when a variable is used in a load context (e.g. `foo`
  in `bar = foo.baz`);
- `pyc.before_binop` / `pyc.after_binop`, `pyc.before_unaryop` /
  `pyc.after_unaryop`, emitted around binary (e.g. `x + y`) and unary (e.g. `-x`,
  `not x`) operations;
- `pyc.after_assign_rhs`, emitted after the right-hand side of an assignment;
- literal events like `pyc.after_int` / `pyc.after_float` / `pyc.after_string`;
- `pyc.call` and `pyc.return_`, two non-AST trace events built in to Python.

Every handler is passed four positional arguments:

1. the return value, for instrumented expressions;
2. the AST node (or node id, if using `register_raw_handler(...)`, or `None`, for
   `sys` events);
3. the stack frame, at the point where instrumentation kicks in;
4. the event (useful when the same handler is registered for multiple events).

Some events pass additional keyword arguments, but the above four suffice for
most use cases — hence the ubiquitous `def handle(self, ret, *_, **__)` shape.

**Handlers can override behavior, not just observe it.** For many events, the
value a handler returns *replaces* the value of the instrumented expression:

```python
class IncrementEveryAssignment(pyc.BaseTracer):
    @pyc.after_assign_rhs
    def handle(self, ret, *_, **__):
        return ret + 1


with IncrementEveryAssignment:
    env = pyc.exec("x = 42")
    assert env["x"] == 43
```

Returning `None` (or nothing) means "don't override." To *actually* override a
value with `None`, return the `pyc.Null` sentinel. Returning `pyc.Skip` stops
further handlers for the current event; `pyc.SkipAll` aborts the whole tracer
stack for that event.

Note that, for AST events, Python source is only transformed to emit an event
when there is at least one active tracer with at least one handler registered for
that event. This keeps the transformed source from becoming bloated when only a
few events are needed.

## A tiny example: exact floats

Because literal events can override the value that flows out of a literal, an
entire behavioral change can fit in a handler. Here's a tracer that makes every
float literal *exact* by promoting it to `Decimal`:

```python
from decimal import Decimal

import pyccolo as pyc


class ExactFloats(pyc.BaseTracer):
    @pyc.after_float
    def to_decimal(self, ret, *_, **__):
        return Decimal(str(ret))


with ExactFloats:
    pyc.exec("print(0.1 + 0.2)")  # -> 0.3   (not 0.30000000000000004)
```

## Composing tracers

A core feature of Pyccolo is that its instrumentation is *composable*. It's
usually tricky to use two or more `ast.NodeTransformer` classes simultaneously
— sometimes you can just have one inherit from the other, but if they both
define `visit` methods for the same AST node type, then typically you would need
to define a bespoke node transformer that uses logic from each base transformer,
handling corner cases to resolve incompatibilities. With Pyccolo, you simply
nest the context managers of each tracer class whose instrumentation you wish to
use, and everything usually Just Works<sup>TM</sup>:

```python
class AddOne(pyc.BaseTracer):
    @pyc.after_assign_rhs
    def handle(self, ret, *_, **__):
        return ret + 1


class TimesTwo(pyc.BaseTracer):
    @pyc.after_assign_rhs
    def handle(self, ret, *_, **__):
        return ret * 2


with AddOne:
    with TimesTwo:
        env = pyc.exec("x = 42")
        assert env["x"] == 86  # (42 + 1) * 2 -- handlers compose in order
```

Return values compose across handlers on the same tracer as well as across
handlers on different tracers.

## Syntax augmentation

Pyccolo can go beyond instrumenting existing Python: a tracer can define *new
surface syntax*. It does this with an `AugmentationSpec`, which declares a
source-level token → replacement rewrite; Pyccolo remembers *where* the rewrite
happened, so a handler can attach to the resulting AST node. For example,
JavaScript-style optional chaining rewrites `?.` down to a plain `.`, then
resolves the access to `None` whenever the receiver is `None`:

```python
optional_chaining_spec = pyc.AugmentationSpec(
    aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
)
```

A complete, tested implementation of optional chaining and nullish coalescing
(`??`) ships in
[pyccolo/examples/optional_chaining.py](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/optional_chaining.py):

```python
import pyccolo as pyc
from pyccolo.examples.optional_chaining import ScriptOptionalChainer

with ScriptOptionalChainer:
    pyc.exec("bar = None\nprint(bar?.foo)")  # -> None
```

The most complete showcase of syntax augmentation is
[**pipescript**](https://github.com/smacke/pipescript), which layers a whole
pipe-and-placeholder dialect on top of Python:

```python
# in IPython / Jupyter, after `%load_ext pipescript`
result = arrays |> map[$
  |> $array[np.isfinite($array)]
  |> np.abs
  |> np.max($, initial=1.0)
] |> max
```

Under the hood, pipescript rewrites illegal token spans like `|>` to legal ones
(here, bitwise-or `|`), then uses Pyccolo to associate the resulting `ast.BinOp`
with the `|>` operator and run the corresponding handler. Because a single
event-emission transform is shared by every handler that cares about it, features
compose without conflicting AST rewrites — see pipescript's ["How it
works"](https://github.com/smacke/pipescript#how-it-works) for the full story.

Syntax augmentation is available on Python >= 3.8. Beyond single-token
replacement, Pyccolo also supports paired-delimiter (brace-block) augmentation
and a `pyc.CustomRewrite` extension point for context-sensitive rewrites; see the
[example gallery](#example-gallery) and
[test_syntax_augmentation.py](https://github.com/smacke/pyccolo/blob/master/test/test_syntax_augmentation.py).

## Source-to-source: `transform`, `untransform`, and pure mode

Sometimes you want the rewritten *source*, not a running program — for a linter,
formatter, or source map. `pyc.transform(code)` returns instrumented / desugared
source, and `pyc.untransform(tree)` reverses an augmentation, resugaring valid
Python back into the augmented syntax:

```python
import pyccolo as pyc
from pyccolo.examples.optional_chaining import ScriptOptionalChainer as OC

with OC:
    # desugar augmented syntax down to plain, valid Python:
    assert pyc.transform("y = a?.b?.c") == "y = a.b.c"

    # ...and resugar it back from the parsed tree:
    tree = pyc.parse("y = a?.b?.c", instrument=False)
    assert pyc.untransform(tree) == "y = a?.b?.c"

    # pure=True marks an analysis-only transform (no runtime side effects):
    assert pyc.transform("y = a?.b", pure=True) == "y = a.b"
```

Both accept a `positions=[(line, col), ...]` argument and return the remapped
positions in the transformed (or untransformed) coordinates, for source-map-style
tooling. Passing `pure=True` signals an analysis-only transform whose result is
never executed; cooperating rewrites can consult `pyc.is_pure_transform()` to
avoid touching execution-relevant state, and it is thread / async-safe via a
`ContextVar`.

## Compatibility with `sys.settrace(...)`

Pyccolo supports not only AST-level instrumentation, but also instrumentation
involving Python's [built-in tracing
utilities](https://docs.python.org/3/library/sys.html#sys.settrace). To use it,
you simply register handlers for one of the corresponding Pyccolo events
(`call`, `line`, `return_`, `exception`, or `opcode`):

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
    with SysTracer:
        def f():
            def g():
                return 42
            return g()
        # push, push, pop, pop
        answer_to_life_universe_everything = f()
```

Note that we didn't need `pyc.exec(...)` here, because Python's built-in tracing
does not involve any AST-level transformations. If we had registered handlers for
AST events such as `pyc.before_stmt`, we *would* need `pyc.exec(...)` to
instrument code in the same file where the tracer is defined.

Pyccolo composes with an existing `sys.settrace(...)` function, too: its unit
tests for `call` and `return_` pass even when
[coverage.py](https://coverage.readthedocs.io/) is active (and without breaking
it), which also uses Python's built-in tracing utilities.

## Instrumenting imported modules

Instrumentation is opt-in for modules imported within tracing contexts. To
determine whether a module gets instrumented, the method
`should_instrument_file(...)` is called with the module's filename as input:

```python
class MyTracer(pyc.BaseTracer):
    def should_instrument_file(self, filename: str) -> bool:
        return filename.endswith("foo.py")

    # handlers, etc. defined below
    ...

with MyTracer:
    import foo  # contents of `foo` get instrumented
    import bar  # contents of `bar` do not
```

Imports are instrumented by registering a custom finder / loader with
`sys.meta_path`. This loader ignores cached bytecode (which may be
uninstrumented), and avoids generating *new* cached bytecode (which would be
instrumented, possibly causing confusion later when instrumentation is not
desired).

## Performance

Pyccolo instrumentation adds significant overhead to Python. In some cases, this
overhead can be partially mitigated if, for example, you only need
instrumentation the first time a statement runs. In such cases, you can
deactivate instrumentation after, e.g., the first time a function executes, or
after the first iteration in a loop, so that further calls (iterations,
respectively) use uninstrumented code with all the mighty performance of native
Python. This is implemented by activating "guards" associated with the function
or loop:

```python
class TracesOnce(pyc.BaseTracer):
    @pyc.register_raw_handler((pyc.after_for_loop_iter, pyc.after_while_loop_iter))
    def after_loop_iter(self, *_, guard, **__):
        self.activate_guard(guard)

    @pyc.register_raw_handler(pyc.after_function_execution)
    def after_function_exec(self, *_, guard, **__):
        self.activate_guard(guard)
```

Subsequent calls / iterations will be instrumented again only after calling
`self.deactivate_guard(...)` on the associated function / loop guard.

## Command line interface

You can execute arbitrary scripts with instrumentation enabled with the `pyc`
command line tool. For example, to use the `OptionalChainer` tracer, given some
example script `bar.py`:

```python
# bar.py
bar = None
# prints `None` since bar?.foo coalesces to `None`
print(bar?.foo)
```

```bash
> pyc bar.py -t pyccolo.examples.OptionalChainer
```

You can also run `bar` as a module (indeed, `pyc` does this internally when given
a file):

```bash
> pyc -m bar -t pyccolo.examples.OptionalChainer
```

You can specify multiple tracer classes after `-t`; in case you were not already
aware, Pyccolo is composable! :)

## Example gallery

Each of the following ships under
[pyccolo/examples/](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples)
as a self-contained, tested tracer — great starting points to adapt:

| Example | Demonstrates |
|---|---|
| [`coverage.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/coverage.py) | statement-level code coverage (`before_stmt`, `should_instrument_file`) |
| [`optional_chaining.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/optional_chaining.py) | `?.`, `.?`, `??` optional chaining / nullish coalescing via `AugmentationSpec` |
| [`pipeline_tracer.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/pipeline_tracer.py) | `\|>` / `\|>>` pipeline operators (binop augmentation) |
| [`quick_lambda.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/quick_lambda.py) | MacroPy-style `f[_ + _]` quick lambdas |
| [`quasiquote.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/quasiquote.py) | MacroPy-style `q[...]` / `u[...]` quasiquotes |
| [`block_lambda.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/block_lambda.py), [`func_block.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/func_block.py), [`brace_subscript.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/brace_subscript.py) | statement-bodied `name{ ... }` blocks (paired-delimiter augmentation) |
| [`lazy_imports.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/lazy_imports.py) | make (most) imports lazy, resolving on first use |
| [`future_tracer.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/future_tracer.py) | implicit async: run assignments on a thread pool, unwrap futures on use |
| [`concolic.py`](https://github.com/smacke/pyccolo/blob/master/pyccolo/examples/concolic.py) | concolic (concrete + symbolic) execution with a Z3 / brute-force solver |

## License

Code in this project licensed under the [BSD-3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
