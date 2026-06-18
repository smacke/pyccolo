# -*- coding: utf-8 -*-
import importlib
from contextlib import ExitStack, contextmanager
from types import CodeType, FunctionType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Set, Type, TypeVar, Union

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


def copy_function_with_code(func: FunctionType, code: CodeType) -> FunctionType:
    """A new function that runs ``code`` but otherwise mirrors ``func``.

    Used by the instrumenter to produce a fresh instrumented function instead of
    rebinding ``func.__code__`` in place. Carries over the metadata that
    ``functools.wraps`` does not (closure, defaults, kwdefaults, dict). The
    ``co_freevars`` guard keeps a recompiled top-level def (which has no free
    vars) from tripping ``FunctionType``'s closure-length check.
    """
    new_func = FunctionType(
        code,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__ if code.co_freevars else None,
    )
    new_func.__kwdefaults__ = func.__kwdefaults__
    new_func.__qualname__ = func.__qualname__
    new_func.__module__ = func.__module__
    new_func.__doc__ = func.__doc__
    new_func.__annotations__ = func.__annotations__
    new_func.__dict__.update(func.__dict__)
    return new_func


K = TypeVar("K")


def clear_keys(d: Union[Dict[K, Any], Set[K]], keys: Iterable[K]) -> None:
    if isinstance(d, dict):
        for key in keys:
            d.pop(key, None)
    elif isinstance(d, set):
        d.difference_update(keys)
    else:
        raise TypeError(f"Unsupported type: {type(d)}")
