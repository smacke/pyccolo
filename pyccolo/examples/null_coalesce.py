# -*- coding: utf-8 -*-
"""
Example of null coalescing implementing with Pyccolo;
  e.g., foo?.bar resolves to `None` when `foo` is `None`.
"""
import ast
import pyccolo as pyc


null_coalesce_spec = pyc.AugmentationSpec(
    aug_type=pyc.AugmentationType.dot, token="?.", replacement="."
)


class NullCoalescer(pyc.BaseTracer):
    class DotIsAlwaysNone:
        def __getattr__(self, _item):
            return None

    dot_is_always_none = DotIsAlwaysNone()

    def should_instrument_file(self, filename: str) -> bool:
        return True

    @property
    def syntax_augmentation_specs(self):
        return [null_coalesce_spec]

    @pyc.register_raw_handler(ast.Attribute)
    def handle_attr_dot(self, ret, node_id, *_, **__):
        if ret is None and null_coalesce_spec in self.get_augmentations(node_id):
            return self.dot_is_always_none
        else:
            return ret
