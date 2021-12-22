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
