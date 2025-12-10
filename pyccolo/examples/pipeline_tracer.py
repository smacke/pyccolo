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
from typing import cast

import pyccolo as pyc

PIPELINE_DOT_PLACEHOLDER_NAME = "__obj"


class HasPipelineDotPlaceholderSpec(ast.NodeVisitor):
    def __init__(self) -> None:
        self._tracer: "PipelineTracer" = None  # type: ignore
        self._found = False

    def visit_Attribute(self, node: ast.Attribute) -> None:
        augmented_nodes = self._tracer.augmented_node_ids_by_spec.get(
            self._tracer.pipeline_dot_placeholder_spec, set()
        )
        if id(node) in augmented_nodes:
            self._found = True
            augmented_nodes.discard(id(node))
        self.generic_visit(node)  # type: ignore

    def __call__(self, node: ast.AST) -> bool:
        self._tracer = PipelineTracer.instance()
        self.visit(node)
        ret = self._found
        self._found = False
        return ret


class PipelineTracer(pyc.BaseTracer):

    ALLOWLIST_BITOR_AS_PIPELINE_OPS_DUNDER_HINT = "__allowlist_bitors_as_pipeline_ops__"

    pipeline_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*|>", replacement="|"
    )

    pipeline_op_assign_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>>", replacement="|"
    )

    pipeline_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
    )

    alt_pipeline_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="%>%", replacement="|"
    )

    value_first_left_partial_apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*$>", replacement="|"
    )

    value_first_left_partial_apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="$>", replacement="|"
    )

    function_first_left_partial_apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<$*", replacement="|"
    )

    function_first_left_partial_apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<$", replacement="|"
    )

    apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<|*", replacement="|"
    )

    apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<|", replacement="|"
    )

    alt_apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="@@", replacement="|"
    )

    compose_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=".* ", replacement="| "
    )

    compose_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=". ", replacement="| "
    )

    pipeline_dot_placeholder_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_prefix,
        token=" .",
        replacement=f" {PIPELINE_DOT_PLACEHOLDER_NAME}.",
    )

    pipeline_dot_placeholder_finder = HasPipelineDotPlaceholderSpec()

    @pyc.register_handler(
        pyc.before_binop,
        when=lambda node: isinstance(node.op, ast.BitOr),
        reentrant=True,
    )
    def handle_before_binop(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        this_node_augmentations = self.get_augmentations(id(node))
        if {
            self.pipeline_op_spec,
            self.alt_pipeline_op_spec,
        } & this_node_augmentations or frame.f_globals.get(
            self.ALLOWLIST_BITOR_AS_PIPELINE_OPS_DUNDER_HINT, False
        ):
            return lambda x, y: y(x)
        elif self.pipeline_tuple_op_spec in this_node_augmentations:
            return lambda x, y: y(*x)
        elif self.compose_op_spec in this_node_augmentations:

            def __pipeline_compose(f, g):
                def __composed(*args, **kwargs):
                    return f(g(*args, **kwargs))

                return __composed

            return __pipeline_compose
        elif self.compose_tuple_op_spec in this_node_augmentations:

            def __pipeline_tuple_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    return f(*g(*args, **kwargs))

                return __tuple_composed

            return __pipeline_tuple_compose
        elif self.pipeline_op_assign_spec in this_node_augmentations:
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
        elif self.value_first_left_partial_apply_op_spec in this_node_augmentations:
            return lambda x, y: (lambda *args: y(x, *args))
        elif (
            self.value_first_left_partial_apply_tuple_op_spec in this_node_augmentations
        ):
            return lambda x, y: (lambda *args: y(*x, *args))
        elif self.function_first_left_partial_apply_op_spec in this_node_augmentations:
            return lambda x, y: (lambda *args: x(y, *args))
        elif (
            self.function_first_left_partial_apply_tuple_op_spec
            in this_node_augmentations
        ):
            return lambda x, y: (lambda *args: x(*y, *args))
        elif {self.apply_op_spec, self.alt_apply_op_spec} & this_node_augmentations:
            return lambda x, y: x(y)
        elif self.apply_tuple_op_spec in this_node_augmentations:
            return lambda x, y: x(*y)
        else:
            return ret

    @pyc.register_handler(
        pyc.before_load_complex_symbol,
        # without this, the trace_lambda can mess up quasiquoting
        when=lambda node: not isinstance(node, ast.Subscript),
        reentrant=True,
    )
    def handle_method_chain(self, ret, node, frame: FrameType, *_, **__):
        if self.pipeline_dot_placeholder_finder(node):
            ret = cast(
                ast.Expr,
                ast.parse(f"lambda {PIPELINE_DOT_PLACEHOLDER_NAME}: None").body[0],
            ).value
            ret.body = node
            return lambda: pyc.eval(ret, frame.f_globals, frame.f_locals)
        else:
            return ret
