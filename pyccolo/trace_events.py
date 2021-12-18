# -*- coding: future_annotations -*-
import sys
from enum import Enum

from pyccolo.utils import fast


class TraceEvent(Enum):
    init_module = 'init_module'

    before_stmt = 'before_stmt'
    after_stmt = 'after_stmt'
    after_module_stmt = 'after_module_stmt'

    load_name = 'load_name'

    before_for_loop_body = 'before_for_loop_body'
    after_for_loop_iter = 'after_for_loop_iter'
    before_while_loop_body = 'before_while_loop_body'
    after_while_loop_iter = 'after_while_loop_iter'

    before_attribute_load = 'before_attribute_load'
    before_attribute_store = 'before_attribute_store'
    before_attribute_del = 'before_attribute_del'
    after_attribute_load = 'after_attribute_load'
    before_subscript_load = 'before_subscript_load'
    before_subscript_store = 'before_subscript_store'
    before_subscript_del = 'before_subscript_del'
    after_subscript_load = 'after_subscript_load'

    subscript_slice = 'subscript_slice'
    _load_saved_slice = '_load_saved_slice'

    before_load_complex_symbol = 'before_complex_symbol'
    after_load_complex_symbol = 'after_complex_symbol'

    before_lambda = 'before_lambda'
    after_lambda = 'after_lambda'

    before_call = 'before_call'
    after_call = 'after_call'
    argument = 'argument'
    before_return = 'before_return'
    after_return = 'after_return'

    before_dict_literal = 'before_dict_literal'
    after_dict_literal = 'after_dict_literal'
    before_list_literal = 'before_list_literal'
    after_list_literal = 'after_list_literal'
    before_set_literal = 'before_set_literal'
    after_set_literal = 'after_set_literal'
    before_tuple_literal = 'before_tuple_literal'
    after_tuple_literal = 'after_tuple_literal'

    dict_key = 'dict_key'
    dict_value = 'dict_value'
    list_elt = 'list_elt'
    set_elt = 'set_elt'
    tuple_elt = 'tuple_elt'

    before_assign_rhs = 'before_assign_rhs'
    after_assign_rhs = 'after_assign_rhs'

    before_function_body = 'before_function_body'
    after_function_execution = 'after_function_execution'

    before_lambda_body = 'before_lambda_body'

    left_binop_arg = 'left_binop_arg'
    right_binop_arg = 'right_binop_arg'
    add = 'add'
    sub = 'sub'
    mult = 'mult'
    mat_mult = 'mat_mult'
    div = 'div'
    floor_div = 'floor_div'
    power = 'power'
    bit_and = 'bit_and'
    bit_or = 'bit_or'
    bit_xor = 'bit_xor'

    ellipses = 'ellipses'

    line = 'line'
    call = 'call'
    return_ = 'return'
    exception = 'exception'
    opcode = 'opcode'

    # these are included for completeness but will probably not be used
    c_call = 'c_call'
    c_return = 'c_return'
    c_exception = 'c_exception'

    def __str__(self):
        return self.value

    def __repr__(self):
        return '<' + str(self) + '>'

    def to_ast(self):
        return fast.Constant(self.value)
