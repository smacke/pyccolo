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

    subscript_slice = "subscript_slice"
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
    add = "add"
    sub = "sub"
    mult = "mult"
    mat_mult = "mat_mult"
    div = "div"
    floor_div = "floor_div"
    power = "power"
    bit_and = "bit_and"
    bit_or = "bit_or"
    bit_xor = "bit_xor"

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

    def to_ast(self):
        return fast.Constant(self.name)


# redundant; these just help with intellisense
init_module = TraceEvent.init_module
before_stmt = TraceEvent.before_stmt
after_stmt = TraceEvent.after_stmt
after_module_stmt = TraceEvent.after_module_stmt
load_name = TraceEvent.load_name
before_for_loop_body = TraceEvent.before_for_loop_body
after_for_loop_iter = TraceEvent.after_for_loop_iter
before_while_loop_body = TraceEvent.before_while_loop_body
after_while_loop_iter = TraceEvent.after_while_loop_iter
before_attribute_load = TraceEvent.before_attribute_load
before_attribute_store = TraceEvent.before_attribute_store
before_attribute_del = TraceEvent.before_attribute_del
after_attribute_load = TraceEvent.after_attribute_load
before_subscript_load = TraceEvent.before_subscript_load
before_subscript_store = TraceEvent.before_subscript_store
before_subscript_del = TraceEvent.before_subscript_del
after_subscript_load = TraceEvent.after_subscript_load
subscript_slice = TraceEvent.subscript_slice
_load_saved_slice = TraceEvent._load_saved_slice
before_load_complex_symbol = TraceEvent.before_load_complex_symbol
after_load_complex_symbol = TraceEvent.after_load_complex_symbol
before_lambda = TraceEvent.before_lambda
after_lambda = TraceEvent.after_lambda
before_call = TraceEvent.before_call
after_call = TraceEvent.after_call
argument = TraceEvent.argument
before_return = TraceEvent.before_return
after_return = TraceEvent.after_return
before_dict_literal = TraceEvent.before_dict_literal
after_dict_literal = TraceEvent.after_dict_literal
before_list_literal = TraceEvent.before_list_literal
after_list_literal = TraceEvent.after_list_literal
before_set_literal = TraceEvent.before_set_literal
after_set_literal = TraceEvent.after_set_literal
before_tuple_literal = TraceEvent.before_tuple_literal
after_tuple_literal = TraceEvent.after_tuple_literal
dict_key = TraceEvent.dict_key
dict_value = TraceEvent.dict_value
list_elt = TraceEvent.list_elt
set_elt = TraceEvent.set_elt
tuple_elt = TraceEvent.tuple_elt
before_assign_rhs = TraceEvent.before_assign_rhs
after_assign_rhs = TraceEvent.after_assign_rhs
before_function_body = TraceEvent.before_function_body
after_function_execution = TraceEvent.after_function_execution
before_lambda_body = TraceEvent.before_lambda_body
left_binop_arg = TraceEvent.left_binop_arg
right_binop_arg = TraceEvent.right_binop_arg
add = TraceEvent.add
sub = TraceEvent.sub
mult = TraceEvent.mult
mat_mult = TraceEvent.mat_mult
div = TraceEvent.div
floor_div = TraceEvent.floor_div
power = TraceEvent.power
bit_and = TraceEvent.bit_and
bit_or = TraceEvent.bit_or
bit_xor = TraceEvent.bit_xor
ellipses = TraceEvent.ellipses
line = TraceEvent.line
call = TraceEvent.call
return_ = TraceEvent.return_
exception = TraceEvent.exception
opcode = TraceEvent.opcode
c_call = TraceEvent.c_call
c_return = TraceEvent.c_return
c_exception = TraceEvent.c_exception


AST_TO_EVENT_MAPPING = {
    ast.stmt: after_stmt,
    ast.Assign: after_assign_rhs,
    ast.Module: init_module,
    ast.Name: load_name,
    ast.Attribute: before_attribute_load,
    ast.Subscript: before_subscript_load,
    ast.Call: before_call,
    ast.Dict: after_dict_literal,
    ast.List: after_list_literal,
    ast.Tuple: after_tuple_literal,
    ast.Set: after_set_literal,
    ast.Return: after_return,
    ast.Add: add,
    ast.Sub: sub,
    ast.Mult: mult,
    ast.Div: div,
    ast.FloorDiv: floor_div,
    ast.MatMult: mat_mult,
    ast.Pow: power,
    ast.BitAnd: bit_and,
    ast.BitOr: bit_or,
    ast.BitXor: bit_xor,
}
