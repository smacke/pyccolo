# -*- coding: future_annotations -*-
from pyccolo._fast.fast_ast import *
from pyccolo._fast.misc_ast_utils import *
import ast


location_of = FastAst.location_of
kw = FastAst.kw
kwargs = FastAst.kwargs
for name in dir(FastAst):
    if hasattr(ast, name):
        globals()[name] = getattr(FastAst, name)
