# -*- coding: utf-8 -*-
import pyccolo as pyc
from pyccolo.examples.brace_subscript import BraceSubscriptTracer
from pyccolo.examples.quick_lambda import QuickLambdaTracer


def test_brace_routes_to_quick_lambda_handler():
    # f{...} should behave exactly like f[...] -- the existing quick_lambda
    # before_subscript_slice handler fires unchanged.
    with QuickLambdaTracer:
        with BraceSubscriptTracer:
            assert pyc.eval("f{_ + _}(3, 4)") == 7
            assert pyc.eval("f{_ * 2}(21)") == 42


def test_brace_multiline_body():
    with QuickLambdaTracer:
        with BraceSubscriptTracer:
            result = pyc.eval(
                """f{
                    _
                    + _
                }(10, 20)"""
            )
    assert result == 30, result


def test_brace_nested():
    with QuickLambdaTracer:
        with BraceSubscriptTracer:
            # nested brace-subscripts: inner f{...} inside outer f{...}
            assert pyc.eval("f{f{_ + 1}(_)}(41)") == 42


def test_brace_and_bracket_equivalent():
    with QuickLambdaTracer:
        with BraceSubscriptTracer:
            assert pyc.eval("f{_ + _}(3, 4)") == pyc.eval("f[_ + _](3, 4)")


class _DetectTracer(pyc.BaseTracer):
    global_guards_enabled = False

    brace_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.subscript,
        token="{",
        replacement="[",
        close_token="}",
        close_replacement="]",
    )

    seen = []

    @pyc.register_handler(pyc.before_subscript_load)
    def _on_sub(self, ret, node, *_, **__):
        self.seen.append(bool(self.brace_spec in self.get_augmentations(id(node))))
        return ret


def test_augmentation_is_detectable():
    # A subscript that came from braces is marked, so a tracer can tell it apart
    # from an ordinary subscript via get_augmentations.
    _DetectTracer.seen = []
    with _DetectTracer:
        # d['a'] is a real subscript (unmarked); d{'a'} -> d['a'] is marked
        _DetectTracer.instance().exec("d = {'a': 1}\nd['a']\nd{'a'}")
    # we expect to have seen at least one marked (from braces) and one unmarked
    assert any(_DetectTracer.seen), _DetectTracer.seen
    assert not all(_DetectTracer.seen), _DetectTracer.seen


def test_normal_braces_untouched():
    with BraceSubscriptTracer:
        assert pyc.eval("{1: 2, 3: 4}") == {1: 2, 3: 4}
        assert pyc.eval("{1, 2, 3}") == {1, 2, 3}
        # keyword + set literal must not be rewritten into a subscript
        assert pyc.exec("def g():\n    return{1, 2}\nout = g()")["out"] == {1, 2}
        # a real subscript still works
        assert pyc.eval("[10, 20, 30][1]") == 20
