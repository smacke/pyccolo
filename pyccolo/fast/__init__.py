# -*- coding: utf-8 -*-
import ast
from pyccolo._fast import *


location_of = FastAst.location_of
kw = FastAst.kw
kwargs = FastAst.kwargs
for name in dir(FastAst):
    if hasattr(ast, name):
        globals()[name] = getattr(FastAst, name)
