# -*- coding: utf-8 -*-
import ast
import copy
import importlib
import logging
import sys
from types import FrameType
from typing import Any, List, Optional, Set, Union

import pyccolo as pyc
from pyccolo._fast.misc_ast_utils import subscript_to_slice
from pyccolo.extra_builtins import PYCCOLO_BUILTIN_PREFIX

logger = logging.getLogger(__name__)


_unresolved = object()


ast_Constant = getattr(ast, "Constant", type(None))
ast_Num = getattr(ast, "Num", type(None))
ast_Str = getattr(ast, "Str", type(None))


class _LazySymbol:
    non_modules: Set[str] = set()
    blocklist_packages: Set[str] = set()

    def __init__(self, spec: Union[ast.Import, ast.ImportFrom]):
        self.spec = spec
        self.value = _unresolved

    @property
    def qualified_module(self) -> str:
        node = self.spec
        name = node.names[0].name
        if isinstance(node, ast.Import):
            return name
        else:
            return f"{node.module}.{name}"

    @staticmethod
    def top_level_package(module: str) -> str:
        return module.split(".", 1)[0]

    @classmethod
    def _unwrap_module(cls, module: str) -> Any:
        if module in sys.modules:
            return sys.modules[module]
        exc = None
        if module not in cls.non_modules:
            try:
                with pyc.allow_reentrant_event_handling():
                    return importlib.import_module(module)
            except ImportError as e:
                cls.non_modules.add(module)
                exc = e
            except Exception:
                logger.error("fatal error trying to import module %s", module)
                raise
        module_symbol = module.rsplit(".", 1)
        if len(module_symbol) != 2:
            raise ValueError("invalid module %s" % module) from exc
        else:
            module, symbol = module_symbol
        ret = getattr(cls._unwrap_module(module), symbol)
        if isinstance(ret, _LazySymbol):
            ret = ret.unwrap()
        return ret

    def _unwrap_helper(self) -> Any:
        return self._unwrap_module(self.qualified_module)

    def unwrap(self) -> Any:
        if self.value is not _unresolved:
            return self.value
        ret = self._unwrap_helper()
        self.value = ret
        return ret

    def __call__(self, *args, **kwargs):
        raise TypeError("cant call _LazyName for spec %s" % ast.unparse(self.spec))

    def __getattr__(self, item):
        raise TypeError(
            "cant __getattr__ on _LazyName for spec %s" % ast.unparse(self.spec)
        )


class _GetLazyNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.lazy_names: Optional[Set[str]] = set()

    def visit_Import(self, node: ast.Import) -> None:
        if self.lazy_names is None:
            return
        for alias in node.names:
            if alias.asname is None:
                return
        for alias in node.names:
            assert alias.asname is not None
            self.lazy_names.add(alias.asname)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self.lazy_names is None:
            return
        for alias in node.names:
            if alias.name == "*":
                self.lazy_names = None
                return
        for alias in node.names:
            self.lazy_names.add(alias.asname or alias.name)

    @classmethod
    def compute(cls, node: ast.Module) -> Set[str]:
        inst = cls()
        inst.visit(node)
        return inst.lazy_names or set()


def _make_attr_guard_helper(node: ast.Attribute) -> Optional[str]:
    if isinstance(node.value, ast.Name):
        return f"{node.value.id}_O_{node.attr}"
    elif isinstance(node.value, ast.Attribute):
        prefix = _make_attr_guard_helper(node.value)
        if prefix is None:
            return None
        else:
            return f"{prefix}_O_{node.attr}"
    else:
        return None


def _make_attr_guard(node: ast.Attribute) -> Optional[str]:
    suffix = _make_attr_guard_helper(node)
    if suffix is None:
        return None
    else:
        return f"{PYCCOLO_BUILTIN_PREFIX}_{suffix}"


def _make_subscript_guard_helper(node: ast.Subscript) -> Optional[str]:
    slice_val = subscript_to_slice(node)
    if isinstance(slice_val, (ast_Constant, ast_Str, ast_Num, ast.Name)):
        if isinstance(slice_val, ast.Name):
            subscript = slice_val.id
        elif hasattr(slice_val, "s"):
            subscript = f"_{slice_val.s}_"  # type: ignore
        elif hasattr(slice_val, "n"):
            subscript = f"_{slice_val.n}_"  # type: ignore
        else:
            return None
    else:
        return None
    if isinstance(node.value, ast.Name):
        return f"{node.value.id}_S_{subscript}"
    elif isinstance(node.value, ast.Subscript):
        prefix = _make_subscript_guard_helper(node.value)
        if prefix is None:
            return None
        else:
            return f"{prefix}_S_{subscript}"
    else:
        return None


def _make_subscript_guard(node: ast.Subscript) -> Optional[str]:
    suffix = _make_subscript_guard_helper(node)
    if suffix is None:
        return None
    else:
        return f"{PYCCOLO_BUILTIN_PREFIX}_{suffix}"


class LazyImportTracer(pyc.BaseTracer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.cur_module_lazy_names: Set[str] = set()
        self.saved_attributes: List[Any] = []
        self.saved_subscripts: List[Any] = []
        self.saved_slices: List[Any] = []

    def _is_name_lazy_load(self, node: Union[ast.Attribute, ast.Name]) -> bool:
        if self.cur_module_lazy_names is None:
            return True
        elif isinstance(node, ast.Name):
            return node.id in self.cur_module_lazy_names
        elif isinstance(node, (ast.Attribute, ast.Subscript)):
            return self._is_name_lazy_load(node.value)  # type: ignore
        elif isinstance(node, ast.Call):
            return self._is_name_lazy_load(node.func)
        else:
            return False

    def static_init_module(self, node: ast.Module) -> None:
        self.cur_module_lazy_names = _GetLazyNames.compute(node)

    @staticmethod
    def _convert_relative_to_absolute(
        package: str, module: Optional[str], level: int
    ) -> str:
        prefix = package.rsplit(".", level - 1)[0]
        if not module:
            return prefix
        else:
            return f"{prefix}.{module}"

    @pyc.init_module
    def init_module(
        self, _ret: None, node: ast.Module, frame: FrameType, *_, **__
    ) -> None:
        assert node is not None
        for guard in self.local_guards_by_module_id.get(id(node), []):
            frame.f_globals[guard] = False

    @pyc.before_stmt(
        when=pyc.Predicate(
            lambda node: isinstance(node, (ast.Import, ast.ImportFrom))
            and pyc.is_outer_stmt(node),
            static=True,
        )
    )
    def before_stmt(
        self,
        _ret: None,
        node: Union[ast.Import, ast.ImportFrom],
        frame: FrameType,
        *_,
        **__,
    ) -> Any:
        is_import = isinstance(node, ast.Import)
        for alias in node.names:
            if alias.name == "*":
                return None
            elif is_import and alias.asname is None:
                return None
        package = frame.f_globals["__package__"]
        level = getattr(node, "level", 0)
        if is_import:
            module = None
        else:
            module = node.module  # type: ignore
            if level > 0:
                module = self._convert_relative_to_absolute(package, module, level)
        for alias in node.names:
            node_cpy = copy.deepcopy(node)
            node_cpy.names = [alias]
            if module is not None:
                node_cpy.module = module  # type: ignore
                node_cpy.level = 0  # type: ignore
            frame.f_globals[alias.asname or alias.name] = _LazySymbol(spec=node_cpy)
        return pyc.Pass

    @pyc.before_attribute_load(
        when=pyc.Predicate(_is_name_lazy_load, static=True), guard=_make_attr_guard
    )
    def before_attr_load(self, ret: Any, *_, **__) -> Any:
        self.saved_attributes.append(ret)
        return ret

    @pyc.after_attribute_load(
        when=pyc.Predicate(_is_name_lazy_load, static=True), guard=_make_attr_guard
    )
    def after_attr_load(
        self, ret: Any, node: ast.Attribute, frame: FrameType, _evt, guard, *_, **__
    ) -> Any:
        if guard is not None:
            frame.f_globals[guard] = True
        saved_attr_obj = self.saved_attributes.pop()
        if isinstance(ret, _LazySymbol):
            ret = ret.unwrap()
            setattr(saved_attr_obj, node.attr, ret)
        return pyc.Null if ret is None else ret

    @pyc.before_subscript_load(
        when=pyc.Predicate(_is_name_lazy_load, static=True), guard=_make_subscript_guard
    )
    def before_subscript_load(self, ret: Any, *_, attr_or_subscript: Any, **__) -> Any:
        self.saved_subscripts.append(ret)
        self.saved_slices.append(attr_or_subscript)
        return ret

    @pyc.after_subscript_load(
        when=pyc.Predicate(_is_name_lazy_load, static=True), guard=_make_subscript_guard
    )
    def after_subscript_load(
        self, ret: Any, _node, frame: FrameType, _evt, guard, *_, **__
    ) -> Any:
        if guard is not None:
            frame.f_globals[guard] = True
        saved_subscript_obj = self.saved_subscripts.pop()
        saved_slice_obj = self.saved_slices.pop()
        if isinstance(ret, _LazySymbol):
            ret = ret.unwrap()
            saved_subscript_obj[saved_slice_obj] = ret
        return pyc.Null if ret is None else ret

    @pyc.load_name(
        when=pyc.Predicate(_is_name_lazy_load, static=True),
        guard=lambda node: f"{PYCCOLO_BUILTIN_PREFIX}_{node.id}_guard",
    )
    def load_name(
        self, ret: Any, node: ast.Name, frame: FrameType, _evt, guard, *_, **__
    ) -> Any:
        if guard is not None:
            frame.f_globals[guard] = True
        if isinstance(ret, _LazySymbol):
            ret = ret.unwrap()
            frame.f_globals[node.id] = ret
        return pyc.Null if ret is None else ret
