# -*- coding: utf-8 -*-
import ast

from pyccolo._fast import (
    EmitterMixin,
    FastAst,
    copy_ast,
    make_composite_condition,
    make_test,
    subscript_to_slice,
)

location_of = FastAst.location_of
kw = FastAst.kw
kwargs = FastAst.kwargs
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
    "kw",
    "kwargs",
]


__all__.extend(name for name in dir(FastAst) if hasattr(ast, name))
