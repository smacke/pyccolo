# -*- coding: utf-8 -*-
import ast
from collections import Counter
from types import FrameType

import pyccolo as pyc


def test_local_guard_activation_prevents_future_handlers():
    class LoadNameCounter(pyc.BaseTracer):
        counter_by_name = Counter()

        @pyc.init_module
        def init_module(self, _ret, node, frame: FrameType, *_, **__):
            assert node is not None
            for guard in self.local_guards_by_module_id.get(id(node), []):
                frame.f_globals[guard] = False

        @pyc.load_name(guard=lambda node: f"_Xix_{node.id}")
        def load_name(
            self, _ret, node: ast.Name, frame: FrameType, _evt, guard, *_, **__
        ):
            self.counter_by_name[node.id] += 1
            assert guard is not None
            frame.f_globals[guard] = True

    with LoadNameCounter.instance():
        pyc.exec(
            """
            w = 0
            x = w + 1
            y = w + x + 1
            z = w + x + y + 1
            """
        )
    for var in ("w", "x", "y"):
        assert LoadNameCounter.counter_by_name[var] == 1, var
