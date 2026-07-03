# -*- coding: utf-8 -*-
import ctypes
import importlib
import inspect
import sys
from contextlib import ExitStack, contextmanager
from types import CodeType, FrameType, FunctionType
from typing import TYPE_CHECKING, Any, Dict, Iterable, Set, Type, TypeVar, Union

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer


def set_frame_local(frame: FrameType, name: str, value: Any) -> None:
    """Assign ``name = value`` in ``frame``'s local scope so a *running* function
    observes the write.

    A frame's ``f_locals`` is normally a read-only view from the outside: on
    CPython < 3.13 it is a snapshot dict, and mutating it does not touch the
    frame's fast-locals array unless ``PyFrame_LocalsToFast`` copies the snapshot
    back. On 3.13+ (PEP 667) ``f_locals`` is a write-through proxy, so assigning
    to it is sufficient. This is the primitive pipescript uses to give macro
    block bodies write-back semantics into their enclosing function.

    Only names that already have a local slot in ``frame`` (function parameters
    and variables assigned somewhere in the enclosing function) are guaranteed to
    persist; ``PyFrame_LocalsToFast`` will not create a brand-new fast slot."""
    frame.f_locals[name] = value
    if sys.version_info < (3, 13):
        ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))


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

    Closure preservation: when ``func`` is a closure but ``code`` was recompiled
    as a top-level def (``inspect.getsource`` yields a bare ``def``/``lambda``, so
    the rewriter emits it at module scope), its free variables are lowered to
    ``LOAD_GLOBAL`` and ``code.co_freevars`` is empty -- the original closure cells
    no longer apply. Resolve those names by layering the captured cell values
    (``getclosurevars(func).nonlocals``) over the module globals, so e.g.
    ``value_and_grad(lambda z: loss(z, targets))`` still sees ``targets`` instead
    of raising ``NameError``. Globals are copied only in this case; an ordinary
    (non-closure) recompile keeps the live ``func.__globals__`` unchanged.
    """
    glbls = func.__globals__
    closure = func.__closure__ if code.co_freevars else None
    if func.__closure__ and not code.co_freevars:
        nonlocals = inspect.getclosurevars(func).nonlocals
        if nonlocals:
            glbls = {**func.__globals__, **nonlocals}
    new_func = FunctionType(
        code,
        glbls,
        func.__name__,
        func.__defaults__,
        closure,
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
