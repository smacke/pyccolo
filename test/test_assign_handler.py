import pytest
import pyccolo as pyc


@pytest.fixture(autouse=True)
def reset_tracer_instance():
    pyc.BaseTracerStateMachine.clear_instance()


def test_sandbox():
    tracer = pyc.BaseTracerStateMachine.instance()
    env = tracer.exec_sandboxed("x = 42")
    assert env["x"] == 42
    assert len(env) == 1


def test_instrumented_sandbox():
    class IncrementsAssignValue(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.TraceEvent.after_assign_rhs)
        def handle_assign(self, ret, *_, **__):
            return ret + 1

    env = IncrementsAssignValue.instance().exec_sandboxed("x = 42")
    assert env["x"] == 43
    assert len(env) == 1


def test_two_handlers():
    class TwoAssignMutations(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.TraceEvent.after_assign_rhs)
        def handle_assign_1(self, ret, *_, **__):
            return ret + 1

        @pyc.register_handler(pyc.TraceEvent.after_assign_rhs)
        def handle_assign_2(self, ret, *_, **__):
            return ret * 2

    env = TwoAssignMutations.instance().exec_sandboxed("x = 42")
    assert env["x"] == 86  # tests that handlers are applied in order of defn
    assert len(env) == 1


def test_null():
    class IncrementsAssignValue(pyc.BaseTracerStateMachine):
        @pyc.register_handler(pyc.TraceEvent.after_assign_rhs)
        def handle_assign_1(self, *_, **__):
            return pyc.Null

        @pyc.register_handler(pyc.TraceEvent.after_assign_rhs)
        def handle_assign_2(self, ret, *_, **__):
            assert ret is None

    env = IncrementsAssignValue.instance().exec_sandboxed("x = 42")
    assert env["x"] is None
    assert len(env) == 1
