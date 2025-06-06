# -*- coding: utf-8 -*-
import ast
import warnings

from pyccolo._fast import (
    EmitterMixin,
    FastAst,
    copy_ast,
    make_composite_condition,
    make_test,
    subscript_to_slice,
)

location_of = FastAst.location_of
location_of_arg = FastAst.location_of_arg
kw = FastAst.kw
kwargs = FastAst.kwargs
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    for name in dir(FastAst):
        if hasattr(ast, name):
            globals()[name] = getattr(FastAst, name)


__all__ = [
    "copy_ast",
    "EmitterMixin",
    "FastAst",
    "make_composite_condition",
    "make_test",
    "subscript_to_slice",
    # now all the ast helper functions
    "location_of",
    "location_of_arg",
    "kw",
    "kwargs",
]


__all__.extend(name for name in dir(FastAst) if hasattr(ast, name))
