# -*- coding: utf-8 -*-
import ast
from enum import Enum

from pyccolo import fast


class TraceEvent(Enum):
    init_module = "init_module"
    import_ = "import"

    before_stmt = "before_stmt"
    after_stmt = "after_stmt"
    after_module_stmt = "after_module_stmt"

    load_name = "load_name"

    before_for_loop_body = "before_for_loop_body"
    after_for_loop_iter = "after_for_loop_iter"
    before_while_loop_body = "before_while_loop_body"
    after_while_loop_iter = "after_while_loop_iter"

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

    before_lambda = "before_lambda"
    after_lambda = "after_lambda"

    before_call = "before_call"
    after_call = "after_call"
    argument = "argument"
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

    before_function_body = "before_function_body"
    after_function_execution = "after_function_execution"

    before_lambda_body = "before_lambda_body"

    left_binop_arg = "left_binop_arg"
    right_binop_arg = "right_binop_arg"
    before_add = "before_add"
    after_add = "after_add"
    before_sub = "before_sub"
    after_sub = "after_sub"
    before_mult = "before_mult"
    after_mult = "after_mult"
    before_mat_mult = "before_mat_mult"
    after_mat_mult = "after_mat_mult"
    before_div = "before_div"
    after_div = "after_div"
    before_mod = "before_mod"
    after_mod = "after_mod"
    before_floor_div = "before_floor_div"
    after_floor_div = "after_floor_div"
    before_power = "before_power"
    after_power = "after_power"
    before_bit_and = "before_bit_and"
    after_bit_and = "after_bit_and"
    before_bit_or = "before_bit_or"
    after_bit_or = "after_bit_or"
    before_bit_xor = "before_bit_xor"
    after_bit_xor = "after_bit_xor"

    ellipses = "ellipses"

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
    TraceEvent.before_add,
    TraceEvent.before_sub,
    TraceEvent.before_mult,
    TraceEvent.before_mat_mult,
    TraceEvent.before_div,
    TraceEvent.before_mod,
    TraceEvent.before_floor_div,
    TraceEvent.before_power,
    TraceEvent.before_bit_and,
    TraceEvent.before_bit_or,
    TraceEvent.before_bit_xor,
    TraceEvent.before_subscript_slice,
}


AST_TO_EVENT_MAPPING = {
    ast.stmt: TraceEvent.after_stmt,
    ast.Assign: TraceEvent.after_assign_rhs,
    ast.Module: TraceEvent.init_module,
    ast.Name: TraceEvent.load_name,
    ast.Attribute: TraceEvent.before_attribute_load,
    ast.Subscript: TraceEvent.before_subscript_load,
    ast.Call: TraceEvent.before_call,
    ast.Dict: TraceEvent.after_dict_literal,
    ast.List: TraceEvent.after_list_literal,
    ast.Tuple: TraceEvent.after_tuple_literal,
    ast.Set: TraceEvent.after_set_literal,
    ast.Return: TraceEvent.after_return,
    ast.Add: TraceEvent.after_add,
    ast.Sub: TraceEvent.after_sub,
    ast.Mult: TraceEvent.after_mult,
    ast.Div: TraceEvent.after_div,
    ast.FloorDiv: TraceEvent.after_floor_div,
    ast.MatMult: TraceEvent.after_mat_mult,
    ast.Pow: TraceEvent.after_power,
    ast.BitAnd: TraceEvent.after_bit_and,
    ast.BitOr: TraceEvent.after_bit_or,
    ast.BitXor: TraceEvent.after_bit_xor,
}
