# -*- coding: utf-8 -*-
import pyccolo as pyc


def test_basic_stack():
    class FunctionTracer(pyc.BaseTracer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stack = self.make_stack()
            with self.stack.register_stack_state():
                with self.stack.needing_manual_initialization():
                    self.name = ""
                self.dummy = None

        def should_propagate_handler_exception(self, _evt, exc: Exception) -> bool:
            return True

        @pyc.register_handler(pyc.before_call)
        def before_call(self, fun, *_, **__):
            with self.stack.push():
                self.name = fun.__name__

        @pyc.register_handler(pyc.after_call)
        def after_call(self, *_, **__):
            self.stack.pop()

    tracer = FunctionTracer.instance()

    with tracer.tracing_enabled():
        # note: everything below is off by 1 because the
        # asserts themselves will call fns that push / pop
        pyc.exec(
            """
            assert tracer.name == ""
            assert len(tracer.stack) == 1
            assert tracer.stack.get_field("name") == ""
            def f():
                assert tracer.name == f.__name__
                assert len(tracer.stack) == 2
                assert tracer.stack.get_field("name") == f.__name__
                assert tracer.stack.get_field("name", depth=2) == ""
                def ggg():
                    assert tracer.name == ggg.__name__
                    assert len(tracer.stack) == 3
                    assert tracer.stack.get_field("name") == ggg.__name__
                    assert tracer.stack.get_field("name", depth=2) == f.__name__
                    assert tracer.stack.get_field("name", depth=3) == ""
                    def hhhhh():
                        assert tracer.name == hhhhh.__name__
                        assert len(tracer.stack) == 4
                        assert tracer.stack.get_field("name") == hhhhh.__name__
                        assert tracer.stack.get_field("name", depth=2) == ggg.__name__
                        assert tracer.stack.get_field("name", depth=3) == f.__name__
                        assert tracer.stack.get_field("name", depth=4) == ""
                    hhhhh()
                    assert tracer.name == ggg.__name__
                    assert len(tracer.stack) == 3
                    assert tracer.stack.get_field("name") == ggg.__name__
                    assert tracer.stack.get_field("name", depth=2) == f.__name__
                    assert tracer.stack.get_field("name", depth=3) == ""
                ggg()
                assert tracer.name == f.__name__
                assert len(tracer.stack) == 2
                assert tracer.stack.get_field("name") == f.__name__
                assert tracer.stack.get_field("name", depth=2) == ""
            f()
            assert tracer.name == ""
            assert len(tracer.stack) == 1
            assert tracer.stack.get_field("name") == ""
            """
        )


class NestedTracer(pyc.BaseTracer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = self.make_stack()
        with self.stack.register_stack_state():
            self.list_stack = self.make_stack()
            with self.list_stack.register_stack_state():
                self.running_length = 0

    def should_propagate_handler_exception(self, _evt, exc: Exception) -> bool:
        return True

    @pyc.register_handler(pyc.before_call)
    def before_call(self, *_, **__):
        with self.stack.push():
            pass

    @pyc.register_handler(pyc.after_call)
    def after_call(self, *_, **__):
        self.stack.pop()

    @pyc.register_handler(pyc.before_list_literal)
    def before_list_literal(self, *_, **__):
        with self.list_stack.push():
            pass

    @pyc.register_handler(pyc.after_list_literal)
    def after_list_literal(self, *_, **__):
        self.list_stack.pop()

    @pyc.register_handler(pyc.list_elt)
    def list_elt(self, *_, **__):
        self.running_length += 1


def test_nested_stack():
    tracer = NestedTracer.instance()
    with tracer.tracing_enabled():
        assert (
            pyc.exec(
                """
                lst = [
                    tracer.running_length,
                    tracer.running_length,
                    [
                        tracer.running_length,
                        tracer.running_length,
                        tracer.running_length,
                    ],
                    tracer.running_length,
                    tracer.running_length,
                ]
                """
            )["lst"]
            == [0, 1, [0, 1, 2], 3, 4]
        )

        assert (
            pyc.exec(
                """
            assert len(tracer.stack) == 1
            def f():
                assert len(tracer.stack) == 2
                return [
                    tracer.running_length,
                    tracer.running_length,
                    [
                        tracer.running_length,
                        tracer.running_length,
                        tracer.running_length,
                    ],
                    tracer.running_length,
                    tracer.running_length,
                ]
            lst = [tracer.running_length, f(), tracer.running_length]
            """
            )["lst"]
            == [0, [0, 1, [0, 1, 2], 3, 4], 2]
        )


def test_clear():
    tracer = NestedTracer.instance()

    def clear_one_level_up():
        tracer.stack.get_field("list_stack").clear()
        return -1

    with tracer.tracing_enabled():
        pyc.exec(
            """
            try:
                lst = [clear_one_level_up()]
            except IndexError:
                pass
            else:
                assert False
            """
        )
