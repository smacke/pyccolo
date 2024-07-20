# -*- coding: utf-8 -*-
import ast
import sys
import textwrap
from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator, List, Optional


class FastAst:
    _LOCATION_OF_NODE: Optional[ast.AST] = None

    if TYPE_CHECKING:

        @staticmethod
        def keyword(arg: str, value: ast.AST) -> ast.keyword:
            ...

    @staticmethod
    @contextmanager
    def location_of(node: ast.AST) -> Generator[None, None, None]:
        """
        All nodes created like `fast.AST(...)` instead of
        `ast.AST(...)` will inherit location info from `node`.
        """
        old_location_of_node = FastAst._LOCATION_OF_NODE
        FastAst._LOCATION_OF_NODE = node
        try:
            yield
        finally:
            FastAst._LOCATION_OF_NODE = old_location_of_node

    @classmethod
    def kw(cls, arg, value) -> ast.keyword:
        return cls.keyword(arg=arg, value=value)

    @classmethod
    def kwargs(cls, **kwargs) -> List[ast.keyword]:
        return [cls.keyword(arg=arg, value=value) for arg, value in kwargs.items()]

    @classmethod
    def parse(cls, code: str, *args, **kwargs) -> ast.AST:
        ret = ast.parse(textwrap.dedent(code), *args, **kwargs)
        if cls._LOCATION_OF_NODE is not None:
            ast.copy_location(ret, cls._LOCATION_OF_NODE)
        return ret

    @classmethod
    def Call(cls, func, args=None, keywords=None, **kwargs) -> ast.Call:
        args = args or []
        keywords = keywords or []
        ret = ast.Call(func, args, keywords, **kwargs)
        if cls._LOCATION_OF_NODE is not None:
            ast.copy_location(ret, cls._LOCATION_OF_NODE)
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
    FastAst.Bytes = staticmethod(_make_func("Constant"))  # type: ignore
    FastAst.NameConstant = staticmethod(_make_func("Constant"))  # type: ignore
    FastAst.Ellipsis = staticmethod(_make_func("Constant"))  # type: ignore
