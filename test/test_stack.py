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

        @pyc.register_handler(pyc.before_call)
        def before_call(self, fun, *_, **__):
            with self.stack.push():
                self.name = fun.__name__

        @pyc.register_handler(pyc.after_call)
        def after_call(self, *_, **__):
            self.stack.pop()

    tracer = FunctionTracer.instance()

    with tracer.tracing_enabled():
        # note: everything below is off by 1 because the asserts themselves will call fns that push / pop
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
