# -*- coding: utf-8 -*-
import importlib
from contextlib import ExitStack, contextmanager
from types import FunctionType
from typing import TYPE_CHECKING, Any, Dict, Type

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer


def resolve_tracer(ref: str) -> Type["BaseTracer"]:
    module, attr = ref.rsplit(".", 1)
    return getattr(importlib.import_module(module), attr)


@contextmanager
def multi_context(cms):
    with ExitStack() as stack:
        yield [stack.enter_context(mgr) for mgr in cms]


def clone_function(func: FunctionType) -> FunctionType:
    local_env: Dict[str, Any] = {}
    exec(
        f"def {func.__name__}(*args, **kwargs): pass",
        func.__globals__,
        local_env,
    )
    cloned_func = local_env[func.__name__]
    cloned_func.__code__ = func.__code__
    return cloned_func
