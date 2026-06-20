# -*- coding: utf-8 -*-
import ast
import sys

import pytest

import pyccolo as pyc

# Syntax-augmentation retention keys nodes by source position, which relies on
# ``end_col_offset``/``end_lineno`` (Python 3.8+). On 3.7 the augmentation never
# maps onto the retained def, so these regressions cannot hold.
requires_end_positions = pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="syntax augmentation retention requires end_col_offset (Python 3.8+)",
)


def test_basic_decorator():
    class IncrementsAssignValue(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    tracer = IncrementsAssignValue.instance()

    @tracer
    def f():
        x = 41
        return x

    assert f() == 42


def test_decorated_tracing_decorator():
    class IncrementsAssignValue(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    tracer = IncrementsAssignValue.instance()

    def twice(f):
        def new_f():
            return f() * 2

        return new_f

    @twice
    @tracer
    def f():
        x = 41
        return x

    assert f() == 84


def test_multiple_tracing_decorators():
    class IncrementsAssignValue1(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    class IncrementsAssignValue2(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 2

    tracer1 = IncrementsAssignValue1.instance()
    tracer2 = IncrementsAssignValue2.instance()

    @pyc.instrumented((tracer1, tracer2))
    def f():
        x = 41
        return x

    assert f() == 44


def _coactive_regression_callee():
    return 42


def test_instrument_for_self_while_other_tracer_active():
    # Regression: instrumenting a function for one tracer while a second tracer
    # is co-active must still weave the instrumenting tracer's events. The
    # instrumenting tracer transiently hard-disables itself during the recompile;
    # it must not be dropped from the rewrite. Here A wants before_call but the
    # co-active B does not, so if A were dropped the call site would get no
    # before_call emit at all and A's interception would silently never fire.
    seen = []

    class WatchesCalls(pyc.BaseTracer):
        @pyc.register_handler(pyc.before_call)
        def handle_call(self, ret, *_, **__):
            seen.append(getattr(ret, "__name__", ret))
            return ret

    class WatchesAssigns(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret

    watch_calls = WatchesCalls.instance()
    watch_assigns = WatchesAssigns.instance()

    with watch_assigns:  # a second tracer is live while we instrument for watch_calls

        @watch_calls
        def f():
            return _coactive_regression_callee()

    assert f() == 42
    assert seen == ["_coactive_regression_callee"]  # before_call fired anyway


def test_instrumented_preserves_sibling_file_bookkeeping():
    # Regression: ``instrumented`` recompiles a *single* function, but its source
    # file may hold other, still-live instrumented code -- most visibly a notebook
    # cell, where one ``co_filename`` is shared by every def/lambda in the cell.
    # Instrumenting one function must not evict the global bookkeeping
    # (``ast_node_by_id``) of the others sharing that file; otherwise their nodes
    # vanish and any runtime node-id lookup against them (e.g. a pipescript ``|>``
    # handler's gate) silently fails and the construct degrades to raw operators.
    class WatchesCalls(pyc.BaseTracer):
        @pyc.register_handler(pyc.before_call)
        def handle_call(self, ret, *_, **__):
            return ret

    tracer = WatchesCalls.instance()

    # Two functions defined in the same file (this test module's co_filename).
    def sibling(y):
        return _coactive_regression_callee()

    def helper(x):
        return x + 1

    assert sibling.__code__.co_filename == helper.__code__.co_filename

    ids_before = set(pyc.BaseTracer.ast_node_by_id)
    tracer.instrumented(sibling)
    sibling_ids = set(pyc.BaseTracer.ast_node_by_id) - ids_before
    assert sibling_ids  # instrumenting sibling registered its nodes

    # Instrumenting another function from the *same file* must not drop them.
    tracer.instrumented(helper)
    assert sibling_ids <= set(pyc.BaseTracer.ast_node_by_id)


@requires_end_positions
def test_instrumented_preserves_augmentations_from_retained_ast():
    # Regression: ``instrumented`` recompiles from ``inspect.getsource``, which yields
    # the *lowered* source (the augmented token is already replaced) -- or no source at
    # all. Re-parsing it loses every syntax augmentation, so a ``when``-gated handler
    # that keys off an augmentation silently stops firing. ``instrumented`` must instead
    # recompile from the retained, augmentation-annotated AST so the markings survive.
    plus_plus = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="++", replacement="+"
    )
    fired = []

    class Add42(pyc.BaseTracer):
        @classmethod
        def syntax_augmentation_specs(cls):
            return [plus_plus]

        @pyc.after_binop(when=lambda node: isinstance(node.op, ast.Add))
        def handle_add(self, ret, node, *_, **__):
            if plus_plus in self.get_augmentations(id(node)):
                fired.append(True)
                return ret + 42
            return ret

    tracer = Add42.instance()
    # Define ``f`` from *surface* source under the tracer: its body BinOp is augmented
    # and the FunctionDef is retained. ``f`` has no recompilable source (a sandbox
    # filename with no linecache entry), so the old getsource path would just raise.
    f = tracer.exec("def f():\n    return 21 ++ 21")["f"]
    assert tracer._augmented_definition_for(f) is not None

    g = tracer.instrumented(f)
    assert g() == 84  # 21 + 21 + 42: the ``++`` augmentation survived the recompile
    assert fired


@requires_end_positions
def test_instrumented_finds_augmented_def_after_sibling_recompiled():
    # Regression: ``ast_bookkeeper_by_fname`` is rebuilt on every recompile to hold only
    # the most recently visited function in a file. Instrumenting one helper must not
    # hide a *sibling* augmented def from the same file (the same notebook cell) from a
    # later ``instrumented`` call. The lookup must consult the global node table, which
    # ``gc_bookkeeping=False`` keeps populated; an earlier version scanned only the
    # per-file bookkeeper and so silently fell back to running siblings un-instrumented.
    plus_plus = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="++", replacement="+"
    )

    class Add42(pyc.BaseTracer):
        @classmethod
        def syntax_augmentation_specs(cls):
            return [plus_plus]

        @pyc.after_binop(when=lambda node: isinstance(node.op, ast.Add))
        def handle_add(self, ret, node, *_, **__):
            if plus_plus in self.get_augmentations(id(node)):
                return ret + 42
            return ret

    tracer = Add42.instance()
    # Two augmented helpers sharing one (cell) file.
    env = tracer.exec("def a():\n    return 1 ++ 1\ndef b():\n    return 2 ++ 2")
    a, b = env["a"], env["b"]
    assert a.__code__.co_filename == b.__code__.co_filename

    # Recompiling ``b`` rebuilds the per-file bookkeeper to hold only ``b``'s nodes...
    tracer.instrumented(b)
    # ...yet ``a``'s augmented def is still found (via the global table) and preserved.
    assert tracer._augmented_definition_for(a) is not None
    assert tracer.instrumented(a)() == 44  # 1 + 1 + 42


def test_instrumented_does_not_mutate_original_by_default():
    class IncrementsAssignValue(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    tracer = IncrementsAssignValue.instance()

    def f():
        x = 41
        return x

    orig_code = f.__code__
    g = tracer.instrumented(f)
    assert g is not f
    assert f.__code__ is orig_code  # original not rewritten in place
    assert g() == 42  # the returned function is instrumented
    assert f() == 41  # the original is left pristine


def test_instrumented_mutate_opt_in_rewrites_original():
    class IncrementsAssignValue(pyc.BaseTracer):
        @pyc.register_handler(ast.Assign)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    tracer = IncrementsAssignValue.instance()

    def f():
        x = 41
        return x

    orig_code = f.__code__
    g = tracer.instrumented(f, mutate=True)
    assert f.__code__ is not orig_code  # mutate=True rewrites in place (old behavior)
    assert g() == 42
