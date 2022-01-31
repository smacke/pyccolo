# -*- coding: utf-8 -*-
import importlib
from contextlib import ExitStack, contextmanager
from typing import Type

from pyccolo.tracer import BaseTracer


def resolve_tracer(ref: str) -> Type[BaseTracer]:
    module, attr = ref.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


@contextmanager
def multi_context(cms):
    with ExitStack() as stack:
        yield [stack.enter_context(mgr) for mgr in cms]
