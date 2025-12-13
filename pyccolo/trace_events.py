# -*- coding: utf-8 -*-
import ast
import sys
import warnings
from enum import Enum

from pyccolo import fast


class TraceEvent(Enum):
    before_import = "before_import"
    init_module = "init_module"
    exit_module = "exit_module"
    after_import = "after_import"

    before_stmt = "before_stmt"
    after_stmt = "after_stmt"
    after_module_stmt = "after_module_stmt"
    after_expr_stmt = "after_expr_stmt"
    _load_saved_expr_stmt_ret = "_load_saved_expr_stmt_ret"

    load_name = "load_name"

    after_bool = "after_bool"
    after_bytes = "after_bytes"
    after_complex = "after_complex"
    after_float = "after_float"
    after_int = "after_int"
    after_none = "after_none"
    after_string = "after_string"

    before_fstring = "before_fstring"
    after_fstring = "after_fstring"

    before_for_loop_body = "before_for_loop_body"
    after_for_loop_iter = "after_for_loop_iter"
    before_while_loop_body = "before_while_loop_body"
    after_while_loop_iter = "after_while_loop_iter"

    before_for_iter = "before_for_iter"
    after_for_iter = "after_for_iter"

    before_attribute_load = "before_attribute_load"
    before_attribute_store = "before_attribute_store"
    before_attribute_del = "before_attribute_del"
    after_attribute_load = "after_attribute_load"
    before_subscript_load = "before_subscript_load"
    before_subscript_store = "before_subscript_store"
    before_subscript_del = "before_subscript_del"
    after_subscript_load = "after_subscript_load"

    before_subscript_slice = "before_subscript_slice"
    after_subscript_slice = "after_subscript_slice"
    _load_saved_slice = "_load_saved_slice"

    before_load_complex_symbol = "before_load_complex_symbol"
    after_load_complex_symbol = "after_load_complex_symbol"

    after_if_test = "after_if_test"
    after_while_test = "after_while_test"

    before_lambda = "before_lambda"
    after_lambda = "after_lambda"

    decorator = "decorator"
    before_call = "before_call"
    after_call = "after_call"
    before_argument = "before_argument"
    after_argument = "after_argument"
    before_return = "before_return"
    after_return = "after_return"

    before_dict_literal = "before_dict_literal"
    after_dict_literal = "after_dict_literal"
    before_list_literal = "before_list_literal"
    after_list_literal = "after_list_literal"
    before_set_literal = "before_set_literal"
    after_set_literal = "after_set_literal"
    before_tuple_literal = "before_tuple_literal"
    after_tuple_literal = "after_tuple_literal"

    dict_key = "dict_key"
    dict_value = "dict_value"
    list_elt = "list_elt"
    set_elt = "set_elt"
    tuple_elt = "tuple_elt"

    before_assign_rhs = "before_assign_rhs"
    after_assign_rhs = "after_assign_rhs"
    before_augassign_rhs = "before_augassign_rhs"
    after_augassign_rhs = "after_augassign_rhs"

    before_assert = "before_assert"
    after_assert = "after_assert"

    before_function_body = "before_function_body"
    after_function_execution = "after_function_execution"

    before_lambda_body = "before_lambda_body"
    after_lambda_body = "after_lambda_body"

    before_left_binop_arg = "before_left_binop_arg"
    after_left_binop_arg = "after_left_binop_arg"
    before_right_binop_arg = "before_right_binop_arg"
    after_right_binop_arg = "after_right_binop_arg"
    before_binop = "before_binop"
    after_binop = "after_binop"

    before_boolop_arg = "before_boolop_arg"
    after_boolop_arg = "after_boolop_arg"
    before_boolop = "before_boolop"
    after_boolop = "after_boolop"

    left_compare_arg = "left_compare_arg"
    compare_arg = "compare_arg"
    before_compare = "before_compare"
    after_compare = "after_compare"

    after_comprehension_if = "after_comprehension_if"
    after_comprehension_elt = "after_comprehension_elt"
    after_dict_comprehension_key = "after_dict_comprehension_key"
    after_dict_comprehension_value = "after_dict_comprehension_value"

    exception_handler_type = "exception_handler_type"

    ellipsis = "ellipsis"

    line = "line"
    call = "call"
    return_ = "return"
    exception = "exception"
    opcode = "opcode"

    # these are included for completeness but will probably not be used
    c_call = "c_call"
    c_return = "c_return"
    c_exception = "c_exception"

    def __str__(self):
        return self.value

    def __repr__(self):
        return "<" + str(self) + ">"

    def __call__(self, handler=None, **kwargs):
        # this will be filled by tracer.py
        ...

    if sys.version_info < (3, 8):

        def to_ast(self):
            return fast.Str(self.name)

    else:

        def to_ast(self):
            return fast.Constant(self.name)


SYS_TRACE_EVENTS = {
    TraceEvent.line,
    TraceEvent.call,
    TraceEvent.return_,
    TraceEvent.exception,
    TraceEvent.opcode,
}


BEFORE_EXPR_EVENTS = {
    TraceEvent.before_argument,
    TraceEvent.before_assign_rhs,
    TraceEvent.before_augassign_rhs,
    TraceEvent.before_binop,
    TraceEvent.before_boolop,
    TraceEvent.before_boolop_arg,
    TraceEvent.before_compare,
    TraceEvent.before_dict_literal,
    TraceEvent.before_for_iter,
    TraceEvent.before_fstring,
    TraceEvent.before_lambda,
    TraceEvent.before_left_binop_arg,
    TraceEvent.before_list_literal,
    TraceEvent.before_load_complex_symbol,
    TraceEvent.before_return,
    TraceEvent.before_right_binop_arg,
    TraceEvent.before_set_literal,
    TraceEvent.before_subscript_slice,
    TraceEvent.before_tuple_literal,
}


AST_TO_EVENT_MAPPING = {
    ast.arg: TraceEvent.after_argument,
    ast.stmt: TraceEvent.after_stmt,
    ast.Assign: TraceEvent.after_assign_rhs,
    ast.Module: TraceEvent.init_module,
    ast.Name: TraceEvent.load_name,
    ast.Attribute: TraceEvent.after_attribute_load,
    ast.Subscript: TraceEvent.after_subscript_load,
    ast.Call: TraceEvent.after_call,
    ast.Dict: TraceEvent.after_dict_literal,
    ast.List: TraceEvent.after_list_literal,
    ast.Tuple: TraceEvent.after_tuple_literal,
    ast.Set: TraceEvent.after_set_literal,
    ast.Return: TraceEvent.after_return,
    ast.BinOp: TraceEvent.after_binop,
    ast.Compare: TraceEvent.after_compare,
}

EVT_TO_EVENT_MAPPING = {
    TraceEvent.before_assert: TraceEvent.before_stmt,
    TraceEvent.after_assert: TraceEvent.after_stmt,
}


with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    ast_Ellipsis = getattr(ast, "Ellipsis", None)
    if ast_Ellipsis is not None:
        AST_TO_EVENT_MAPPING[ast_Ellipsis] = TraceEvent.ellipsis
