# -*- coding: utf-8 -*-
"""Tests for the native IPython integration (``%load_ext pyccolo``).

Each test that drives a shell runs in a **fresh subprocess**: IPython's shell is
a process-wide singleton, and so are pyccolo's tracers, so state leaks across
tests in-process and makes outcomes order-dependent. This mirrors
``pipescript/test/test_reexecution.py``.
"""
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("IPython")


_PREAMBLE = """
import sys
import pyccolo as pyc
from pyccolo.emit_event import _TRACER_STACK
from IPython.testing.globalipapp import get_ipython

ip = get_ipython()

def run(code):
    result = ip.run_cell(code)
    if result.error_in_exec is not None:
        raise AssertionError("cell %r raised %r" % (code, result.error_in_exec))
    return result
"""


def _probe(body: str) -> None:
    """Run ``body`` against a real IPython shell in a subprocess."""
    script = _PREAMBLE + textwrap.dedent(body) + '\nprint("OK")\n'
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_register_before_load_raises():
    _probe(
        """
        class T(pyc.BaseTracer):
            @pyc.before_stmt
            def h(self, *_, **__): pass

        try:
            pyc.register_ipython_tracer(T, shell=ip)
        except RuntimeError as exc:
            assert "load_ext pyccolo" in str(exc), exc
        else:
            raise AssertionError("expected RuntimeError")
        """
    )


def test_instruments_cells_and_survives_reexecution():
    _probe(
        """
        run("%load_ext pyccolo")
        run("import pyccolo as pyc\\nhits = []\\n"
            "class T(pyc.BaseTracer):\\n"
            "    @pyc.before_stmt\\n"
            "    def h(self, *_, **__): hits.append(1)\\n")
        run("%pyccolo register T")

        before = len(ip.user_ns["hits"])
        run("a = 1\\nb = a + 1")
        assert len(ip.user_ns["hits"]) - before == 2, ip.user_ns["hits"]
        assert ip.user_ns["b"] == 2

        # re-executing the same cell must instrument again (ast bookkeeping)
        before = len(ip.user_ns["hits"])
        run("a = 1\\nb = a + 1")
        assert len(ip.user_ns["hits"]) - before == 2
        """
    )


def test_out_is_preserved_for_instrumented_cells():
    """A ``before_stmt`` handler must not cost the user their ``Out[N]``."""
    _probe(
        """
        run("%load_ext pyccolo")
        run("import pyccolo as pyc\\n"
            "class T(pyc.BaseTracer):\\n"
            "    @pyc.before_stmt\\n"
            "    def h(self, *_, **__): pass\\n")
        run("%pyccolo register T")

        assert run("1 + 1").result == 2
        assert run("x = 2\\nx + 3").result == 5
        assert run("y = 4").result is None
        """
    )


def test_sys_trace_tracer_and_exceptions():
    """A ``call`` handler installs ``sys.settrace``; exceptions must still work."""
    _probe(
        """
        run("%load_ext pyccolo")
        run("import pyccolo as pyc\\ncalls = []\\n"
            "class T(pyc.BaseTracer):\\n"
            "    @pyc.call\\n"
            "    def h(self, _ret, _node, frame, *_a, **_k):\\n"
            "        calls.append(frame.f_code.co_name)\\n")
        run("%pyccolo register T")

        assert run("def f():\\n    return 7\\nf()").result == 7
        assert "f" in ip.user_ns["calls"]

        result = ip.run_cell("raise ValueError('boom')")
        assert isinstance(result.error_in_exec, ValueError), result.error_in_exec

        run("%pyccolo deregister all")
        assert sys.gettrace() is None
        """
    )


def test_registration_order_is_canonical():
    _probe(
        """
        run("%load_ext pyccolo")
        run("import pyccolo as pyc\\norder = []\\n"
            "class A(pyc.BaseTracer):\\n"
            "    @pyc.before_stmt\\n"
            "    def h(self, *_, **__): order.append('A')\\n"
            "class B(pyc.BaseTracer):\\n"
            "    @pyc.before_stmt\\n"
            "    def h(self, *_, **__): order.append('B')\\n")
        run("%pyccolo register A")
        run("%pyccolo register B")

        names = [c.__name__ for c in pyc.registered_ipython_tracers(shell=ip)]
        assert names == ["A", "B"], names

        del ip.user_ns["order"][:]
        run("pass")
        # first registered sees the event first
        assert ip.user_ns["order"][:2] == ["A", "B"], ip.user_ns["order"]

        stack = [type(t).__name__ for t in _TRACER_STACK]
        assert stack == ["A", "B"], stack
        """
    )


def test_all_four_tracer_spellings():
    _probe(
        """
        run("%load_ext pyccolo")
        run("import pyccolo as pyc\\n"
            "class T(pyc.BaseTracer):\\n"
            "    @pyc.before_stmt\\n"
            "    def h(self, *_, **__): pass\\n")
        T = ip.user_ns["T"]

        def only_registered():
            return [c.__name__ for c in pyc.registered_ipython_tracers(shell=ip)]

        # 1. class
        pyc.register_ipython_tracer(T, shell=ip)
        assert only_registered() == ["T"]
        run("%pyccolo deregister all")

        # 2. instance
        pyc.register_ipython_tracer(T.instance(), shell=ip)
        assert only_registered() == ["T"]
        run("%pyccolo deregister all")

        # 3. bare user-namespace name
        run("%pyccolo register T")
        assert only_registered() == ["T"]
        run("%pyccolo deregister all")

        # 4. qualified path (a cell-defined tracer lives in __main__)
        pyc.register_ipython_tracer("__main__.T", shell=ip)
        assert only_registered() == ["T"]

        # ...and a genuinely importable one
        run("%pyccolo deregister all")
        pyc.register_ipython_tracer("pyccolo.tracer.NoopTracer", shell=ip)
        assert only_registered() == ["NoopTracer"]
        """
    )


def test_cell_magics_are_not_syntax_transformed():
    _probe(
        """
        run("%load_ext pyccolo")
        run("import pyccolo as pyc\\n"
            "class T(pyc.BaseTracer):\\n"
            "    @pyc.before_stmt\\n"
            "    def h(self, *_, **__): pass\\n")
        run("%pyccolo register T")
        run("%%capture captured\\nq = 5\\n")
        assert ip.user_ns["q"] == 5
        """
    )


def test_syntax_augmented_cell_runs_without_execution_info_transformed_cell():
    """A host whose ``ExecutionInfo`` has no ``transformed_cell`` (IPython < 9)
    must still get the source phase's rewriter, and with it the AST rewrite.

    Without it ``|>`` is augmented to a bare ``|`` and then never rewritten into
    a pipeline call, so the cell silently evaluates a bitwise-or instead.
    """
    _probe(
        """
        from IPython.core.interactiveshell import ExecutionInfo

        # Emulate IPython < 9, where ``ExecutionInfo`` carries no transformed cell.
        _orig_init = ExecutionInfo.__init__

        def _init(self, *args, **kwargs):
            kwargs.pop("transformed_cell", None)
            _orig_init(self, *args, **kwargs)
            self.__dict__.pop("transformed_cell", None)

        ExecutionInfo.__init__ = _init
        if "transformed_cell" in vars(ExecutionInfo):
            del ExecutionInfo.transformed_cell

        run("%load_ext pyccolo")
        run("from pyccolo.examples import PipelineTracer")
        run("%pyccolo register PipelineTracer")

        assert not hasattr(ExecutionInfo("x", False, False, True, None), "transformed_cell")
        assert run("(1, 2, 3) |> list").result == [1, 2, 3]
        assert run("(3, 4, 1) |> sorted |> tuple").result == (1, 3, 4)
        """
    )


def test_stale_transform_is_not_adopted_by_a_later_cell():
    """The ``transform_cell`` fallback is keyed on the raw cell, so a recorded
    transform from some other source (e.g. a completion) is never mistaken for
    the cell ``pre_run_cell`` is announcing."""
    _probe(
        """
        run("%load_ext pyccolo")
        driver = pyc.ipython_driver(ip)

        class _Info:
            def __init__(self, raw_cell):
                self.raw_cell = raw_cell

        driver._last_transform = ("a + 1", "a + 1\\n")
        assert driver._transformed_cell_for(_Info("a + 1"), "a + 1") == "a + 1\\n"
        assert driver._transformed_cell_for(_Info("b + 2"), "b + 2") is None

        driver._last_transform = None
        assert driver._transformed_cell_for(_Info("a + 1"), "a + 1") is None

        # A host that does supply it always wins over the recorded fallback.
        class _Info9(_Info):
            transformed_cell = "from_host\\n"

        driver._last_transform = ("a + 1", "recorded\\n")
        assert driver._transformed_cell_for(_Info9("a + 1"), "a + 1") == "from_host\\n"
        """
    )


def test_unload_restores_the_shell():
    _probe(
        """
        orig_ast = list(ip.ast_transformers)
        orig_input = list(ip.input_transformers_post)
        orig_cache = ip.compile.cache
        orig_transform_cell = ip.transform_cell

        run("%load_ext pyccolo")
        run("import pyccolo as pyc\\nhits = []\\n"
            "class T(pyc.BaseTracer):\\n"
            "    @pyc.before_stmt\\n"
            "    def h(self, *_, **__): hits.append(1)\\n")
        run("%pyccolo register T")
        run("a = 1")
        assert len(ip.user_ns["hits"]) > 0

        run("%unload_ext pyccolo")
        before = len(ip.user_ns["hits"])
        run("b = 2")
        assert len(ip.user_ns["hits"]) == before, "tracer still firing after unload"

        assert ip.ast_transformers == orig_ast, ip.ast_transformers
        assert ip.input_transformers_post == orig_input, ip.input_transformers_post
        assert ip.compile.cache == orig_cache
        assert ip.transform_cell == orig_transform_cell
        assert sys.gettrace() is None
        assert not any(
            type(f).__name__ == "TraceFinder" for f in sys.meta_path
        ), sys.meta_path
        assert _TRACER_STACK == [], _TRACER_STACK
        assert "pyccolo" not in ip.magics_manager.magics["line"]
        """
    )


def test_extension_module_stays_jupyterlite_safe():
    """The module must not drag in anything absent under Pyodide."""
    script = textwrap.dedent(
        """
        import sys
        import pyccolo.ipython_extension  # noqa: F401

        forbidden = [m for m in ("ipykernel", "zmq", "comm") if m in sys.modules]
        assert not forbidden, forbidden
        # Importing pyccolo must not even require IPython to be importable.
        assert "IPython" not in sys.modules
        print("OK")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"
