# -*- coding: utf-8 -*-
"""
Implementation of a pipeline operators in Pyccolo -- including a purely functional `|>` operator,
as well as an assigning operator `|>>`, to showcase syntax augmentation capabilities for binops.

Example:
```
with PipelineTracer:
    pyc.eval("(1, 2, 3) |> list")
>>> [1, 2, 3]
```

If we want to assign the result:
```
with PipelineTracer:
    pyc.exec("(1, 2, 3) |> list |>> result; print(result)")
>>> [1, 2, 3]
```

PipelineTracer is especially effective when combined with QuickLambdaTracer:
```
with PipelineTracer:
    with QuickLambdaTracer:
        pyc.exec("(1, 2, 3) |> list |>> result |> f[map(f[_ + 1], _)] |> list |> f[print(_, end=' ')]; print(result)")
>>> [2, 3, 4] [1, 2, 3]
```
"""
import ast
from types import FrameType

import pyccolo as pyc


class PipelineTracer(pyc.BaseTracer):

    ALLOWLIST_BITOR_AS_PIPELINE_OPS_DUNDER_HINT = "__allowlist_bitors_as_pipeline_ops__"

    pipeline_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
    )

    pipeline_op_assign_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>>", replacement="|"
    )

    @property
    def syntax_augmentation_specs(self):
        return [self.pipeline_op_assign_spec, self.pipeline_op_spec]

    @pyc.register_handler(
        pyc.before_binop,
        when=lambda node: isinstance(node.op, ast.BitOr),
        reentrant=True,
    )
    def handle_before_binop(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        if self.pipeline_op_spec in self.get_augmentations(
            id(node)
        ) or frame.f_globals.get(
            self.ALLOWLIST_BITOR_AS_PIPELINE_OPS_DUNDER_HINT, False
        ):
            return lambda x, y: y(x)
        elif self.pipeline_op_assign_spec in self.get_augmentations(id(node)):
            rhs: ast.Name = node.right  # type: ignore
            if not isinstance(rhs, ast.Name):
                raise ValueError(
                    "unable to assign to RHS of type %s" % type(node.right)
                )
            # eagerly assign it so that we don't get a name error
            frame.f_globals[rhs.id] = None

            def assign_globals(val):
                frame.f_globals[rhs.id] = val
                return val

            return lambda x, y: assign_globals(x)
        else:
            return ret
