# -*- coding: utf-8 -*-
import pyccolo as pyc


def test_basic_instrumented_import():
    class IncrementsAssignValue(pyc.BaseTracer):
        def should_instrument_file(self, filename: str) -> bool:
            return filename.endswith("foo.py")

        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign(self, ret, node, *_, **__):
            node_id = id(node)
            assert self.ast_node_by_id[node_id] is node
            assert node_id in self.containing_ast_by_id
            assert node_id in self.containing_stmt_by_id
            return ret + 1

    with IncrementsAssignValue.instance().tracing_enabled():
        import test.foo  # noqa
