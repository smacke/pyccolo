# -*- coding: utf-8 -*-
import ast
import sys
from types import FrameType

import pyccolo as pyc


def test_sandbox():
    env = pyc.exec("x = 42", local_env={})
    assert env["x"] == 42
    assert len(env) == 1, "got %s" % env

    env = pyc.exec("locals()['x'] = 43", local_env={})
    assert env["x"] == 43
    assert len(env) == 1, "got %s" % env


def test_instrumented_sandbox():
    class IncrementsAssignValue(pyc.BaseTracer):
        @pyc.after_assign_rhs
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    env = IncrementsAssignValue.instance().exec("x = 42", local_env={})
    assert env["x"] == 43, "got %s" % env["x"]
    assert len(env) == 1


def test_nonlocal_1():
    x = 5  # noqa
    try:
        pyc.exec("nonlocal x; x = 42")
    except SyntaxError:
        # we impl this by executing in a sandboxed function;
        # the 'nonlocal' keyword will conflict with function arg
        pass
    else:
        assert False


def test_nonlocal_2():
    x = 5

    def inner():
        nonlocal x
        x = pyc.exec("x = 42")["x"]

    inner()

    assert x == 42


def test_two_handlers():
    class TwoAssignMutations(pyc.BaseTracer):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_1(self, ret, *_, **__):
            return ret + 1

        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_2(self, ret, *_, **__):
            return ret * 2

    with TwoAssignMutations.instance().tracing_context():
        env = pyc.exec("x = 42", local_env={})
    # env = TwoAssignMutations.instance().exec("x = 42")
    assert env["x"] == 86, (
        "got %s" % env["x"]
    )  # tests that handlers are applied in order of defn
    assert len(env) == 1


def test_two_handlers_from_separate_classes():
    class AssignMutation1(pyc.BaseTracer):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_1(self, ret, *_, **__):
            return ret + 1

    class AssignMutation2(pyc.BaseTracer):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_2(self, ret, *_, **__):
            return ret * 2

    with AssignMutation1.instance():
        with AssignMutation2.instance():
            env = pyc.exec("x = 42", local_env={})

    assert env["x"] == 86, (
        "got %s" % env["x"]
    )  # tests that handlers are applied in order of defn
    assert len(env) == 1


def test_null():
    class IncrementsAssignValue(pyc.BaseTracer):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_1(self, *_, **__):
            return pyc.Null

        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_2(self, ret, *_, **__):
            assert ret is None

    env = IncrementsAssignValue.instance().exec("x = 42", local_env={})
    assert env["x"] is None
    assert len(env) == 1


def test_pass_sandboxed_environ():
    env = pyc.exec("x = 42", local_env={})
    assert env["x"] == 42
    assert len(env) == 1, "got %s" % env
    env = pyc.exec("y = x + 1", local_env=env)
    assert len(env) == 2
    assert env["x"] == 42
    assert env["y"] == 43


def test_sys_tracing_call():
    class TracesCalls(pyc.BaseTracer):
        def __init__(self):
            super().__init__()
            self.num_calls_seen = 0

        @pyc.register_handler(pyc.call)
        def handle_call(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name in ("foo", "bar"):
                self.num_calls_seen += 1

        @pyc.register_handler(pyc.return_)
        def handle_return(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name in ("foo", "bar"):
                assert self.num_calls_seen >= 1

    sys_tracer = sys.gettrace()
    tracer = TracesCalls.instance()
    assert tracer.has_sys_trace_events
    env = tracer.exec(
        """
        def foo():
            pass
        foo(); foo()
        """,
        local_env={},
    )
    assert sys.gettrace() is sys_tracer
    assert len(env) == 1
    assert "foo" in env
    assert tracer.num_calls_seen == 2

    tracer.num_calls_seen = 0
    with TracesCalls.instance().tracing_disabled():
        pyc.exec(
            """
            def foo():
                with TracesCalls.instance().tracing_enabled():
                    def bar():
                        pass
                    bar()
            foo(); foo()
            """
        )
    assert sys.gettrace() is sys_tracer
    assert tracer.num_calls_seen == 2


def test_composed_sys_tracing_calls():
    original_tracer = sys.gettrace()
    num_calls_seen_1 = 0
    num_calls_seen_2 = 0
    num_calls_seen_3 = 0

    class TracesCalls1(pyc.BaseTracer):
        @pyc.register_handler(pyc.call)
        def handle_call(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name == "foo":
                nonlocal num_calls_seen_1
                num_calls_seen_1 += 1
                assert num_calls_seen_1 > num_calls_seen_2
                assert num_calls_seen_1 > num_calls_seen_3

        @pyc.register_handler(pyc.return_)
        def handle_return(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name == "foo":
                assert num_calls_seen_1 >= 1

    class TracesCalls2(pyc.BaseTracer):
        @pyc.register_handler(pyc.call)
        def handle_call(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name == "foo":
                nonlocal num_calls_seen_2
                num_calls_seen_2 += 1
                assert num_calls_seen_2 > num_calls_seen_3

        @pyc.register_handler(pyc.return_)
        def handle_return(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name == "foo":
                assert num_calls_seen_2 >= 1

    class TracesCalls3(pyc.BaseTracer):
        @pyc.register_handler(pyc.call)
        def handle_call(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name == "foo":
                nonlocal num_calls_seen_3
                num_calls_seen_3 += 1

        @pyc.register_handler(pyc.return_)
        def handle_return(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name == "foo":
                assert num_calls_seen_3 >= 1

    with TracesCalls1.instance().tracing_context():
        with TracesCalls2.instance().tracing_context():
            with TracesCalls3.instance().tracing_context():
                env = pyc.exec(
                    """
                    def foo():
                        pass
                    foo(); foo()
                    """,
                    local_env={},
                )
    assert sys.gettrace() is original_tracer
    assert len(env) == 1
    assert "foo" in env
    assert num_calls_seen_1 == 2
    assert num_calls_seen_2 == 2
    assert num_calls_seen_3 == 2

    num_calls_seen_1 = num_calls_seen_2 = num_calls_seen_3 = 0
    # now make sure it works with the package-level one

    with pyc.tracing_context(
        (TracesCalls1.instance(), TracesCalls2.instance(), TracesCalls3.instance())
    ):
        env = pyc.exec(
            """
            def foo():
                pass
            foo(); foo()
            """,
            local_env={},
        )
    assert sys.gettrace() is original_tracer
    assert len(env) == 1
    assert "foo" in env
    assert num_calls_seen_1 == 2
    assert num_calls_seen_2 == 2
    assert num_calls_seen_3 == 2


def test_reentrant_handling():
    class AssignMutationOuter(pyc.BaseTracer):
        def should_instrument_file(self, filename: str) -> bool:
            return "test" in filename

        @pyc.register_handler(ast.Assign)
        def handle_outer_assign(self, ret, _node, frame, *_, **__):
            if type(ret) != int or frame.f_code.co_filename != "<sandbox>":
                return
            return ret + 2

    code = """
        class AssignMutationInner(pyc.BaseTracer):
            @pyc.register_handler(ast.Assign)
            def handle_inner_assign(self, ret, *_, **__):
                if not isinstance(ret, int):
                    return
                with inner_tracer.tracing_disabled():
                    new_ret = ret + 1  # + 2 if reentrant else + 0
                return new_ret
        inner_tracer = AssignMutationInner.instance()
        with outer_tracer.tracing_disabled():
            with inner_tracer.tracing_disabled():
                x = pyc.exec(
                    '''
                    with outer_tracer.tracing_enabled():
                        with inner_tracer.tracing_enabled():
                            x = 37  # + 5 if reentrant else + 3
                    '''
                )["x"]
        """

    outer_tracer = AssignMutationOuter.instance()

    with outer_tracer.tracing_enabled():
        assert pyc.exec(code)["x"] == 40

    with pyc.allow_reentrant_event_handling():
        with outer_tracer.tracing_enabled():
            assert pyc.exec(code)["x"] == 42


def test_tracing_context_manager_toggling():
    num_stmts_seen = 0
    num_calls_seen = 0

    class TracesStatements(pyc.BaseTracer):
        @pyc.register_handler(ast.stmt)
        def handle_stmt(self, _ret, _node, frame, *_, **__):
            if frame.f_code.co_filename == "<sandbox>":
                nonlocal num_stmts_seen
                num_stmts_seen += 1

    class TracesCalls(pyc.BaseTracer):
        @pyc.register_handler(pyc.call)
        def handle_call(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name == "foo":
                nonlocal num_calls_seen
                num_calls_seen += 1

    stmt_tracer = TracesStatements()
    call_tracer = TracesCalls()

    with stmt_tracer.tracing_enabled():
        with stmt_tracer.tracing_disabled():
            with call_tracer.tracing_enabled():
                pyc.exec(
                    """
                    def foo():
                        return 5
                    foo()
                """
                )
        pyc.exec("x = 1")

    assert num_stmts_seen == 1
    assert num_calls_seen == 1

    num_stmts_seen = num_calls_seen = 0
    with pyc.tracing_disabled((stmt_tracer, call_tracer)):
        pyc.exec(
            """
            def foo():
                return 5
            foo()
            x = 1
            """
        )
    assert num_stmts_seen == 0
    assert num_calls_seen == 0

    with pyc.tracing_enabled((stmt_tracer, call_tracer)):
        pyc.exec(
            """
            def foo():
                return 5
            foo()
            x = 1
            """
        )

    assert num_stmts_seen == 3
    assert num_calls_seen == 1


def test_composes_with_existing_sys_tracer():

    num_calls_seen_from_existing_tracer = 0

    sys_settrace = sys.settrace
    # in case we are using codecov
    prev_sys_tracer = sys.gettrace()

    def existing_tracer(frame: FrameType, evt, *args, **kwargs):
        if (
            evt == "call"
            and frame.f_code.co_filename == "<sandbox>"
            and frame.f_code.co_name in ("foo", "bar")
        ):
            nonlocal num_calls_seen_from_existing_tracer
            num_calls_seen_from_existing_tracer += 1
        if prev_sys_tracer is not None:
            cur_sys_tracer = sys.gettrace()
            ret = prev_sys_tracer(frame, evt, *args, **kwargs)
            if sys.gettrace() != cur_sys_tracer:
                sys_settrace(cur_sys_tracer)
            if ret == prev_sys_tracer:
                return existing_tracer
            else:
                return ret

    class TracesCalls(pyc.BaseTracer):
        def __init__(self):
            super().__init__()
            self.num_calls_seen = 0

        @pyc.register_handler(pyc.call)
        def handle_call(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name in ("foo", "bar"):
                self.num_calls_seen += 1

        @pyc.register_handler(pyc.return_)
        def handle_return(self, _ret, _node, frame: FrameType, *_, **__):
            if frame.f_code.co_name in ("foo", "bar"):
                assert self.num_calls_seen >= 1

    tracer = TracesCalls.instance()
    try:
        # the top one tests that sys.settrace gets patched properly
        # even when the first enabled tracer doesn't do anything
        # with sys events
        with pyc.BaseTracer():
            with tracer.tracing_disabled():
                pyc.exec(
                    """
                    def foo():
                        with TracesCalls.instance().tracing_enabled():
                            with TracesCalls.instance().tracing_disabled():
                                with TracesCalls.instance().tracing_enabled():
                                    sys.settrace(existing_tracer)
                                    def bar():
                                        pass
                                    bar()
                    foo(); foo()
                    """
                )
        assert sys.gettrace() is existing_tracer
    finally:
        sys.settrace(prev_sys_tracer)
    assert tracer.num_calls_seen == 2
    assert num_calls_seen_from_existing_tracer == 3


def test_override_stmt():
    for assertion_val in (True, False):

        class OverridesStmt(pyc.BaseTracer):
            @pyc.before_stmt
            def handle_before_stmt(self, *_, **__):
                return f"globals()['x'] = 43; assert {assertion_val}"

            @pyc.after_stmt
            def handle_after_stmt(self, *_, **__):
                globals()["x"] = 44

        with OverridesStmt.instance():
            try:
                pyc.exec("pass")
            except AssertionError:
                assert x == 43
            else:
                assert x == 44
    del globals()["x"]


def test_before_call():
    class OverridesCall(pyc.BaseTracer):
        @pyc.before_call
        def before_call(self, ret, *_, **__):
            return lambda: 42

    assert (
        OverridesCall.instance().exec(
            """
        def foo():
            return 43
        x = foo()
        """
        )["x"]
        == 42
    )


def test_before_add_returning_callable():
    class OverridesAdd(pyc.BaseTracer):
        @pyc.before_add
        def before_add(self, *_, **__):
            return lambda x, y: (x + y) % 42

        @pyc.after_add
        def after_add(self, ret, *_, **__):
            assert ret == 1
            return ret + 1

    with OverridesAdd.instance():
        assert (
            pyc.exec(
                """
            x = 41 + 2
            """
            )["x"]
            == 2
        )


def test_before_add_returning_constant():
    class OverridesAdd(pyc.BaseTracer):
        @pyc.before_add
        def before_add(self, *_, **__):
            return 41

        @pyc.after_add
        def after_add(self, ret, *_, **__):
            assert ret == 41
            return ret + 1

    with OverridesAdd.instance():
        assert (
            pyc.exec(
                """
            x = 41 + 4
            """
            )["x"]
            == 42
        )


def test_skip():
    class SkipsSecondHandler(pyc.BaseTracer):
        @pyc.before_add
        def before_add(self, *_, **__):
            return 41

        @pyc.after_add
        def after_add(self, ret, *_, **__):
            assert ret == 41
            return pyc.Skip

        @pyc.after_add
        def skipped_after_add(self, ret, *_, **__):
            return ret + 1

    assert SkipsSecondHandler.instance().exec("x = 41 + 4")["x"] == 41


def test_conditional_instrumentation():
    class IncrementsX(pyc.BaseTracer):
        @pyc.load_name(when=lambda node: node.id == "x")
        def load_x(self, ret, *_, **__):
            return ret + 1

    with IncrementsX.instance():
        d = pyc.exec(
            """
            x = 0
            y = 0
            a = x
            b = y
            c = a + b + x + y
            """
        )
    assert d["a"] == 1
    assert d["b"] == 0
    assert d["c"] == 2


def test_quasiquotes():
    from pyccolo.examples import Quasiquoter

    with Quasiquoter.instance():
        nested = pyc.eval("q[u[str(q[42])]]")
        binop = pyc.eval("q[41 + 1]")
        d = pyc.exec(
            """
            a = 10
            b = 2
            lst = [1, 2, 3]
            x = lst[-1]
            node1 = q[a + b]
            node2 = q[1 + u[a + b]]
            node3 = q["a" + "b"]
            node4 = q[name["a"] + "b"]
            node5 = q[ast_literal[node3] + ast_literal[node4]]
            node6 = q[ast_list[node1, node2, node3, node4, node5]]
            """
        )
    assert nested.s.startswith("<ast.") or nested.s.startswith("<_ast.")

    assert isinstance(binop, ast.BinOp)
    assert isinstance(binop.op, ast.Add)
    assert binop.left.n == 41
    assert binop.right.n == 1

    assert d["x"] == d["lst"][-1] == 3
    node1, node2, node3, node4, node5, node6 = (
        d["node1"],
        d["node2"],
        d["node3"],
        d["node4"],
        d["node5"],
        d["node6"],
    )
    assert isinstance(node1, ast.BinOp), "got %s" % type(node1)
    assert isinstance(node2, ast.BinOp), "got %s" % type(node2)
    assert node2.left.n == 1
    assert node2.right.n == 12

    assert node3.left.s == "a"
    assert node3.right.s == "b"
    assert isinstance(node4.left, ast.Name) and node4.left.id == "a"
    assert node4.right.s == "b"

    assert isinstance(node5, ast.BinOp)
    assert isinstance(node5.op, ast.Add)
    assert ast.dump(node5.left) == ast.dump(node3)
    assert ast.dump(node5.right) == ast.dump(node4)

    assert isinstance(node6, ast.List)
    for elt, expected in zip(node6.elts, [node1, node2, node3, node4, node5]):
        assert ast.dump(elt) == ast.dump(expected)


def test_quick_lambda():
    from pyccolo.examples import QuickLambdaTracer, Quasiquoter

    Quasiquoter.clear_instance()
    with QuickLambdaTracer.instance():
        assert pyc.eval("f[_ + _](41, 1)") == 42
        assert pyc.eval("f[_ + f[_ * _](3, 4)](1)") == 13
