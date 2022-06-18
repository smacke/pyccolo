# -*- coding: utf-8 -*-
"""
Example of null coalescing implementing with Pyccolo;
  e.g., foo?.bar resolves to `None` when `foo` is `None`.
"""
import pyccolo as pyc

optional_chaining_spec = pyc.AugmentationSpec(
    aug_type=pyc.AugmentationType.dot, token="?.", replacement="."
)


class OptionalChainer(pyc.BaseTracer):
    class ResolvesToNone:
        def __getattr__(self, _item):
            return None

        def __call__(self, *_, **__):
            return None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lexical_call_stack: pyc.TraceStack = self.make_stack()
        with self.lexical_call_stack.register_stack_state():
            # TODO: pop this the right number of times if an exception occurs
            self.cur_call_is_none_resolver: bool = False

    resolves_to_none = ResolvesToNone()

    def should_instrument_file(self, filename: str) -> bool:
        return True

    @property
    def syntax_augmentation_specs(self):
        return [optional_chaining_spec]

    @pyc.register_raw_handler(pyc.after_module_stmt)
    def handle_after_module_stmt(self, *_, **__):
        while len(self.lexical_call_stack) > 0:
            self.lexical_call_stack.pop()

    @pyc.register_raw_handler(pyc.before_attribute_load)
    def handle_before_attr(self, obj, node_id, *_, **__):
        if obj is None and optional_chaining_spec in self.get_augmentations(node_id):
            return self.resolves_to_none
        else:
            return obj

    @pyc.register_raw_handler(pyc.before_call)
    def handle_before_call(self, func, *_, **__):
        with self.lexical_call_stack.push():
            self.cur_call_is_none_resolver = func is self.resolves_to_none

    @pyc.register_raw_handler(pyc.before_argument)
    def handle_before_arg(self, arg_lambda, *_, **__):
        if self.cur_call_is_none_resolver:
            return lambda: None
        else:
            return arg_lambda

    @pyc.register_raw_handler(pyc.after_call)
    def handle_after_call(self, *_, **__):
        self.lexical_call_stack.pop()

    @pyc.register_raw_handler(pyc.after_attribute_load)
    def handle_after_attr(self, ret, node_id, *_, call_context, **__):
        if (
            ret is None
            and optional_chaining_spec in self.get_augmentations(node_id)
            and call_context
        ):
            return self.resolves_to_none
        else:
            return ret
