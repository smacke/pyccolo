# -*- coding: utf-8 -*-
import ast
import functools
import sys
import textwrap
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Generator, List, Optional

from typing_extensions import TypeVar

T = TypeVar("T", bound=ast.AST)


class FastAst:
    _LOCATION_OF_NODE: Optional[ast.AST] = None

    if TYPE_CHECKING:

        @staticmethod
        def keyword(arg: str, value: ast.AST) -> ast.keyword: ...
        @staticmethod
        def Str(*args, **kwargs) -> ast.Constant: ...
        @staticmethod
        def Num(*args, **kwargs) -> ast.Constant: ...
        @staticmethod
        def Bytes(*args, **kwargs) -> ast.Constant: ...
        @staticmethod
        def NameConstant(*args, **kwargs) -> ast.Constant: ...
        @staticmethod
        def Ellipsis(*args, **kwargs) -> ast.Constant: ...

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
    def location_of_arg(cls, func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapped_node_transform(*args) -> T:
            with cls.location_of(args[-1]):
                return func(*args)

        return wrapped_node_transform

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

    @staticmethod
    def iter_arguments(args: ast.arguments) -> Generator[ast.arg, None, None]:
        yield from getattr(args, "posonlyargs", [])
        yield from args.args
        if args.vararg is not None:
            yield args.vararg
        yield from getattr(args, "kwonlyargs", [])
        if args.kwarg is not None:
            yield args.kwarg


def _make_func(new_name, old_name=None):
    def ctor(*args, **kwargs):
        ret = getattr(ast, new_name)(*args, **kwargs)
        if FastAst._LOCATION_OF_NODE is not None:
            ast.copy_location(ret, FastAst._LOCATION_OF_NODE)
        return ret

    ctor.__name__ = old_name or new_name
    return ctor


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    for ctor_name in ast.__dict__:
        if ctor_name.startswith("_") or hasattr(FastAst, ctor_name):
            continue
        setattr(FastAst, ctor_name, staticmethod(_make_func(ctor_name)))


if sys.version_info >= (3, 8):
    for old_name in ("Str", "Num", "Bytes", "NameConstant", "Ellipsis"):
        setattr(FastAst, old_name, staticmethod(_make_func("Constant", old_name)))
