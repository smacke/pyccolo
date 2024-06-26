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

        @pyc.load_name(guard=lambda node: f"{pyc.PYCCOLO_BUILTIN_PREFIX}_{node.id}")
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


def test_subscript_local_guard_activation():
    class SubscriptCounter(pyc.BaseTracer):
        counter_by_subscript = Counter()

        @pyc.init_module
        def init_module(self, _ret, node: ast.Module, frame: FrameType, *_, **__):
            assert node is not None
            for guard in self.local_guards_by_module_id.get(id(node), []):
                frame.f_globals[guard] = False

        @pyc.before_subscript_load(
            guard=lambda node: f"{pyc.PYCCOLO_BUILTIN_PREFIX}_{node.value.id}"
        )
        def before_subscript_load(self, _ret, node, *_, **__):
            self.counter_by_subscript[node.value.id] += 1

        @pyc.after_subscript_load(
            guard=lambda node: f"{pyc.PYCCOLO_BUILTIN_PREFIX}_{node.value.id}"
        )
        def after_subscript_load(
            self, _ret, node, frame: FrameType, _evt, guard, *_, **__
        ):
            self.counter_by_subscript[node.value.id] += 1
            assert guard is not None
            frame.f_globals[guard] = True

    with SubscriptCounter.instance():
        pyc.exec(
            """
            lst = [0]
            x = lst[0] + 1
            y = lst[0] + x + 1
            z = lst[0] + x + y + 1
            assert z == 4
            """
        )
    for var in ("lst",):
        assert SubscriptCounter.counter_by_subscript[var] == 2, var
