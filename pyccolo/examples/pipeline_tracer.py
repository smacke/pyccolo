# -*- coding: utf-8 -*-
import ast

import pyccolo as pyc


class PipelineTracer(pyc.BaseTracer):

    pipelining_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
    )

    @property
    def syntax_augmentation_specs(self):
        return [self.pipelining_spec]

    @pyc.register_handler(
        pyc.before_binop, when=lambda node: isinstance(node.op, ast.BitOr)
    )
    def handle_before_binop(self, ret: object, node: ast.BinOp, *_, **__):
        if self.pipelining_spec not in self.get_augmentations(id(node)):
            return ret
        return lambda x, y: y(x)
