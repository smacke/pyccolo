# -*- coding: utf-8 -*-
"""
Example of optional chaining and nullish coalescing implementing with Pyccolo;
  e.g., foo?.bar resolves to `None` when `foo` is `None`.
"""
import ast
from typing import Any, Optional

import pyccolo as pyc


def parent_is_or_boolop(node_id: int) -> bool:
    parent = pyc.BaseTracer.containing_ast_by_id.get(node_id)
    return isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.Or)


class OptionalChainer(pyc.BaseTracer):
    class ResolvesToNone:
        def __init__(self, eventually: bool) -> None:
            self.__eventually = eventually

        def __getattr__(self, _item: str):
            if self.__eventually:
                return self
            else:
                return None

        def __call__(self, *_, **__):
            return self

    resolves_to_none_eventually = ResolvesToNone(eventually=True)
    resolves_to_none_immediately = ResolvesToNone(eventually=False)

    call_optional_chaining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token="?.(", replacement="("
    )

    optional_chaining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
    )

    permissive_attr_dereference_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token=".?", replacement="."
    )

    nullish_coalescing_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.boolop, token="??", replacement=" or "
    )

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved_ret_expr = None
        self.lexical_call_stack: pyc.TraceStack = self.make_stack()
        with self.lexical_call_stack.register_stack_state():
            # TODO: pop this the right number of times if an exception occurs
            self.cur_call_is_none_resolver: bool = False
        self.lexical_nullish_stack: pyc.TraceStack = self.make_stack()
        with self.lexical_nullish_stack.register_stack_state():
            self.cur_boolop_has_nullish_coalescer = False
            self.coalesced_value: Optional[Any] = None

    @pyc.register_raw_handler(pyc.after_stmt)
    def handle_after_stmt(self, ret, *_, **__):
        self._saved_ret_expr = ret

    @pyc.register_raw_handler(pyc.after_module_stmt)
    def handle_after_module_stmt(self, *_, **__):
        while len(self.lexical_call_stack) > 0:
            self.lexical_call_stack.pop()
        ret = self._saved_ret_expr
        self._saved_ret_expr = None
        return ret

    @pyc.register_handler(pyc.before_attribute_load)
    def handle_before_attr(self, obj, node: ast.Attribute, *_, **__):
        if (
            self.optional_chaining_spec in self.get_augmentations(id(node))
            and obj is None
        ):
            return self.resolves_to_none_eventually
        elif self.permissive_attr_dereference_spec in self.get_augmentations(
            id(node)
        ) and not hasattr(obj, node.attr):
            return self.resolves_to_none_immediately
        else:
            return obj

    @pyc.register_handler(pyc.before_call)
    def handle_before_call(self, func, node: ast.Call, *_, **__):
        if func is None and self.call_optional_chaining_spec in self.get_augmentations(
            id(node.func)
        ):
            func = self.resolves_to_none_eventually
        with self.lexical_call_stack.push():
            self.cur_call_is_none_resolver = func is self.resolves_to_none_eventually
        return func

    @pyc.register_raw_handler(pyc.before_argument)
    def handle_before_arg(self, arg_lambda, *_, **__):
        if self.cur_call_is_none_resolver:
            return lambda: None
        else:
            return arg_lambda

    @pyc.register_raw_handler(pyc.after_call)
    def handle_after_call(self, *_, **__):
        self.lexical_call_stack.pop()

    @pyc.register_raw_handler(pyc.after_load_complex_symbol)
    def handle_after_load_complex_symbol(self, ret, *_, **__):
        if isinstance(ret, self.ResolvesToNone):
            return pyc.Null
        else:
            return ret

    @pyc.register_handler(
        pyc.before_boolop, when=lambda node: isinstance(node.op, ast.Or)
    )
    def before_or_boolop(self, ret, node: ast.BoolOp, *_, **__):
        with self.lexical_nullish_stack.push():
            self.cur_boolop_has_nullish_coalescer = any(
                self.nullish_coalescing_spec in self.get_augmentations(id(val))
                for val in node.values
            )
            self.coalesced_value = None
        return ret

    @pyc.register_handler(
        pyc.after_boolop, when=lambda node: isinstance(node.op, ast.Or)
    )
    def after_or_boolop(self, *_, **__):
        self.lexical_nullish_stack.pop()

    def _maybe_compute_nullish_coalesced_value(self, ret, node_id: int) -> None:
        if self.coalesced_value is not None:
            return
        val = ret()
        if self.nullish_coalescing_spec in self.get_augmentations(node_id):
            self.coalesced_value = None if val is None else val
        else:
            self.coalesced_value = val or None

    @pyc.register_raw_handler(pyc.before_boolop_arg, when=parent_is_or_boolop)
    def before_or_boolup(self, ret, node_id: int, *_, is_last: bool, **__):
        if self.cur_boolop_has_nullish_coalescer:
            if is_last:
                if self.coalesced_value is None:
                    return ret
                else:
                    return lambda: self.coalesced_value
            else:
                self._maybe_compute_nullish_coalesced_value(ret, node_id)
                return lambda: None
        else:
            return ret


class ScriptOptionalChainer(OptionalChainer):
    def should_instrument_file(self, filename: str) -> bool:
        return True
