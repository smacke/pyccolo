import sys
import pyccolo as pyc


def test_sandbox():
    env = pyc.exec("x = 42")
    assert env["x"] == 42
    assert len(env) == 1, 'got %s' % env


def test_instrumented_sandbox():
    class IncrementsAssignValue(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    env = IncrementsAssignValue.instance().exec("x = 42")
    assert env["x"] == 43, 'got %s' % env["x"]
    assert len(env) == 1


def test_two_handlers():
    class TwoAssignMutations(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_1(self, ret, *_, **__):
            return ret + 1

        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_2(self, ret, *_, **__):
            return ret * 2

    with TwoAssignMutations.instance().tracing_context():
        env = pyc.exec("x = 42")
    # env = TwoAssignMutations.instance().exec("x = 42")
    assert env["x"] == 86, 'got %s' % env["x"]  # tests that handlers are applied in order of defn
    assert len(env) == 1


def test_two_handlers_from_separate_classes():
    class AssignMutation1(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_1(self, ret, *_, **__):
            return ret + 1

    class AssignMutation2(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_2(self, ret, *_, **__):
            return ret * 2

    with AssignMutation1.instance().tracing_context():
        with AssignMutation2.instance().tracing_context():
            env = pyc.exec("x = 42")

    assert env["x"] == 86, 'got %s' % env["x"]  # tests that handlers are applied in order of defn
    assert len(env) == 1


def test_null():
    class IncrementsAssignValue(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_1(self, *_, **__):
            return pyc.Null

        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign_2(self, ret, *_, **__):
            assert ret is None

    env = IncrementsAssignValue.instance().exec("x = 42")
    assert env["x"] is None
    assert len(env) == 1


def test_pass_sandboxed_environ():
    env = pyc.exec("x = 42")
    assert env["x"] == 42
    assert len(env) == 1, 'got %s' % env
    env = pyc.exec("y = x + 1", local_env=env)
    assert len(env) == 2
    assert env["x"] == 42
    assert env["y"] == 43


def test_sys_tracing_call():
    num_calls_seen = 0

    class TracesCalls(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.call)
        def handle_call(self, *_, **__):
            nonlocal num_calls_seen
            num_calls_seen += 1

        @pyc.register_handler(pyc.return_)
        def handle_return(self, *_, **__):
            assert num_calls_seen >= 1

    assert TracesCalls.instance().has_sys_trace_events
    env = TracesCalls.instance().exec(
        """
        def foo():
            pass
        foo(); foo()
        """
    )
    assert len(env) == 1
    assert "foo" in env
    assert num_calls_seen == 2


def test_composed_sys_tracing_calls():
    num_calls_seen_1 = 0
    num_calls_seen_2 = 0
    num_calls_seen_3 = 0

    class TracesCalls1(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.call)
        def handle_call(self, *_, **__):
            nonlocal num_calls_seen_1
            num_calls_seen_1 += 1
            assert num_calls_seen_1 > num_calls_seen_2
            assert num_calls_seen_1 > num_calls_seen_3

        @pyc.register_handler(pyc.return_)
        def handle_return(self, *_, **__):
            assert num_calls_seen_1 >= 1

    class TracesCalls2(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.call)
        def handle_call(self, *_, **__):
            nonlocal num_calls_seen_2
            num_calls_seen_2 += 1
            assert num_calls_seen_2 > num_calls_seen_3

        @pyc.register_handler(pyc.return_)
        def handle_return(self, *_, **__):
            assert num_calls_seen_2 >= 1

    class TracesCalls3(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.call)
        def handle_call(self, *_, **__):
            nonlocal num_calls_seen_3
            num_calls_seen_3 += 1

        @pyc.register_handler(pyc.return_)
        def handle_return(self, *_, **__):
            assert num_calls_seen_3 >= 1

    with TracesCalls1.instance().tracing_context():
        with TracesCalls2.instance().tracing_context():
            with TracesCalls3.instance().tracing_context():
                env = pyc.exec(
                    """
                    def foo():
                        pass
                    foo(); foo()
                    """
                )
    assert sys.gettrace() is None
    assert len(env) == 1
    assert "foo" in env
    assert num_calls_seen_1 == 2
    assert num_calls_seen_2 == 2
    assert num_calls_seen_3 == 2
