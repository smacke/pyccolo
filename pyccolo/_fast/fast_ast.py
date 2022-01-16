# -*- coding: utf-8 -*-
import ast
import sys
import textwrap
from contextlib import contextmanager


class FastAst:
    _LOCATION_OF_NODE = None

    @staticmethod
    @contextmanager
    def location_of(node):
        """
        All nodes created like `fast.AST(...)` instead of
        `ast.AST(...)` will inherit location info from `node`.
        """
        old_location_of_node = FastAst._LOCATION_OF_NODE
        FastAst._LOCATION_OF_NODE = node
        yield
        FastAst._LOCATION_OF_NODE = old_location_of_node

    @staticmethod
    def kw(arg, value):
        return FastAst.keyword(arg=arg, value=value)

    @staticmethod
    def kwargs(**kwargs):
        return [FastAst.keyword(arg=arg, value=value) for arg, value in kwargs.items()]

    @staticmethod
    def parse(code, *args, **kwargs):
        ret = ast.parse(textwrap.dedent(code, *args, **kwargs))
        if FastAst._LOCATION_OF_NODE is not None:
            ast.copy_location(ret, FastAst._LOCATION_OF_NODE)
        return ret

    @staticmethod
    def Call(func, args=None, keywords=None, **kwargs):
        args = args or []
        keywords = keywords or []
        ret = ast.Call(func, args, keywords, **kwargs)
        if FastAst._LOCATION_OF_NODE is not None:
            ast.copy_location(ret, FastAst._LOCATION_OF_NODE)
        return ret


def _make_func(func_name):
    def ctor(*args, **kwargs):
        ret = getattr(ast, func_name)(*args, **kwargs)
        if FastAst._LOCATION_OF_NODE is not None:
            ast.copy_location(ret, FastAst._LOCATION_OF_NODE)
        return ret

    return ctor


for ctor_name in ast.__dict__:
    if ctor_name.startswith("_") or hasattr(FastAst, ctor_name):
        continue
    setattr(FastAst, ctor_name, staticmethod(_make_func(ctor_name)))

if sys.version_info >= (3, 8):
    FastAst.Str = staticmethod(_make_func("Constant"))  # type: ignore
    FastAst.Num = staticmethod(_make_func("Constant"))  # type: ignore
