# -*- coding: utf-8 -*-
import pyccolo as pyc
from pyccolo.examples.func_block import FuncBlockTracer


def test_run_statement_body():
    with FuncBlockTracer:
        result = pyc.eval(
            """run{
                total = 0
                for i in range(5):
                    total += i
                total
            }"""
        )
    assert result == 10, result


def test_run_expression_body():
    with FuncBlockTracer:
        assert pyc.eval("run{ 2 + 3 }") == 5


def test_thunk_is_deferred():
    with FuncBlockTracer:
        f = pyc.eval(
            """thunk{
                xs = []
                for i in range(3):
                    xs.append(i * i)
                xs
            }"""
        )
    assert callable(f)
    assert f() == [0, 1, 4]


def test_run_closes_over_locals():
    with FuncBlockTracer:
        n = 6  # noqa: F841  (closed over by the run{...} block below)
        result = pyc.eval(
            """run{
                acc = 1
                for i in range(1, n):
                    acc *= i
                acc
            }"""
        )
    assert result == 120, result  # 5!


def test_nested_blocks():
    with FuncBlockTracer:
        result = pyc.eval(
            """run{
                evens = run{
                    out = []
                    for i in range(6):
                        if i % 2 == 0:
                            out.append(i)
                    out
                }
                sum(evens)
            }"""
        )
    assert result == 6, result  # 0 + 2 + 4


def test_passes_a_real_callable_through_subscript_handler():
    # Demonstrate the slice really is a function value: a pyccolo
    # after_subscript_slice handler receives the freshly-defined callable.
    class _Probe(pyc.BaseTracer):
        global_guards_enabled = False
        block_spec = pyc.AugmentationSpec(
            aug_type=pyc.AugmentationType.subscript,
            token="{",
            replacement="[",
            close_token="}",
            close_replacement="]",
            name_pattern="grab",
            body_func_wrapper="__pyc_fn__",
        )
        captured = []

        def enter_tracing_hook(self):
            import builtins

            builtins.__pyc_fn__ = FuncBlockTracer.instance().__pyc_fn__
            builtins.grab = type("G", (), {"__getitem__": lambda self, x: x})()

        def exit_tracing_hook(self):
            import builtins

            for name in ("__pyc_fn__", "grab"):
                if hasattr(builtins, name):
                    delattr(builtins, name)

        @pyc.register_handler(pyc.after_subscript_slice, reentrant=True)
        def _grab(self, ret, node, *_, **__):
            if self.block_spec in self.get_augmentations(id(node)):
                type(self).captured.append(ret)
            return ret

    _Probe.captured = []
    with FuncBlockTracer:  # provides __pyc_fn__ via its instance
        with _Probe:
            val = pyc.eval("grab{ x = 7\n x * 6 }")
    assert callable(_Probe.captured[-1]), _Probe.captured
    assert val() == 42, val


def test_other_names_untouched():
    with FuncBlockTracer:
        # names not in the macro pattern keep brace meaning (set/dict literal)
        assert pyc.eval("{1, 2, 3}") == {1, 2, 3}
        assert pyc.eval("{'a': 1}") == {"a": 1}
