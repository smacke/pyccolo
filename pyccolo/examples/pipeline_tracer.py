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
import builtins
import itertools
from contextlib import contextmanager
from types import FrameType
from typing import Any, Callable, Dict, Generator, Optional, Set, Union, cast

import pyccolo as pyc
from pyccolo.examples.optional_chaining import OptionalChainer
from pyccolo.stmt_mapper import StatementMapper


class ExtractNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if node.id != "_":
            self.names.add(node.id)

    def generic_visit_comprehension(
        self, node: Union[ast.GeneratorExp, ast.DictComp, ast.ListComp, ast.SetComp]
    ) -> None:
        before_names = set(self.names)
        self.generic_visit(node)
        after_names = self.names
        self.names = set()
        for gen in node.generators:
            self.visit(gen.target)
        # need to clear names referenced as targets since these
        # do not need to be passed externally to any lambdas
        for name in self.names:
            if name not in before_names:
                after_names.discard(name)
        self.names = after_names

    visit_GeneratorExp = visit_DictComp = visit_ListComp = visit_SetComp = (
        generic_visit_comprehension
    )

    @classmethod
    def extract_names(cls, node: ast.expr) -> Set[str]:
        visitor = cls()
        visitor.visit(node)
        return visitor.names


class SingletonArgCounterMixin:
    _arg_ctr = 0

    @property
    def arg_ctr(self) -> int:
        return self._arg_ctr

    @arg_ctr.setter
    def arg_ctr(self, new_arg_ctr: int) -> None:
        SingletonArgCounterMixin._arg_ctr = new_arg_ctr

    @classmethod
    def create_placeholder_lambda(
        cls, orig_ctr: int, lambda_body: ast.expr, frame_globals: Dict[str, Any]
    ) -> ast.Lambda:
        num_lambda_args = cls._arg_ctr - orig_ctr
        extra_defaults = ExtractNames.extract_names(lambda_body)
        lambda_args = []
        for arg_idx in range(orig_ctr, orig_ctr + num_lambda_args):
            arg = f"_{arg_idx}"
            lambda_args.append(arg)
            extra_defaults.discard(arg)
        extra_defaults = {
            arg
            for arg in extra_defaults
            if arg not in frame_globals and not hasattr(builtins, arg)
        }
        lambda_arg_str = ", ".join(
            itertools.chain(lambda_args, (f"{arg}={arg}" for arg in extra_defaults))
        )
        return cast(
            ast.Lambda,
            cast(ast.Expr, ast.parse(f"lambda {lambda_arg_str}: None").body[0]).value,
        )


class PlaceholderReplacer(ast.NodeVisitor, SingletonArgCounterMixin):
    def __init__(self):
        self.mutate = False
        self.allow_top_level = False

    @contextmanager
    def disallow_top_level(self) -> Generator[None, None, None]:
        old_allow_top_level = self.allow_top_level
        try:
            self.allow_top_level = False
            yield
        finally:
            self.allow_top_level = old_allow_top_level

    def visit_Call(self, node: ast.Call) -> None:
        self.visit(node.func)
        if not self.allow_top_level:
            # defer visiting nested calls
            return
        with self.disallow_top_level():
            for arg in node.args:
                self.visit(arg)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        from pyccolo.examples.quick_lambda import QuickLambdaTracer

        if (
            not self.allow_top_level
            and isinstance(node.value, ast.Name)
            and node.value.id in QuickLambdaTracer.lambda_macros
        ):
            # defer visiting nested quick lambdas
            return
        with self.disallow_top_level():
            self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if (
            not self.allow_top_level
            and isinstance(node.op, ast.BitOr)
            and PipelineTracer.get_augmentations(id(node))
        ):
            # defer visiting nested pipeline ops
            return
        with self.disallow_top_level():
            self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id != "_":
            return
        if (
            id(node)
            not in PipelineTracer.augmented_node_ids_by_spec[
                PipelineTracer.arg_placeholder_spec
            ]
        ):
            return
        if self.mutate:
            node.id = f"_{self.arg_ctr}"
            PipelineTracer.augmented_node_ids_by_spec[
                PipelineTracer.arg_placeholder_spec
            ].discard(id(node))
        self.arg_ctr += 1

    def search(self, node: ast.expr, allow_top_level: bool) -> bool:
        orig_ctr = self.arg_ctr
        try:
            self.allow_top_level = allow_top_level
            self.visit(node)
            found = self.arg_ctr > orig_ctr
        finally:
            self.arg_ctr = orig_ctr
        return found

    def rewrite(self, node: ast.expr, allow_top_level: bool) -> None:
        old_mutate = self.mutate
        try:
            self.mutate = True
            self.allow_top_level = allow_top_level
            self.visit(node)
        finally:
            self.mutate = old_mutate


def parent_is_bitor_op(node: ast.expr) -> bool:
    parent = pyc.BaseTracer.containing_ast_by_id.get(id(node))
    return isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.BitOr)


class PipelineTracer(pyc.BaseTracer):

    pipeline_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="**|>", replacement="|"
    )

    pipeline_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*|>", replacement="|"
    )

    pipeline_op_assign_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>>", replacement="|"
    )

    pipeline_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="|>", replacement="|"
    )

    value_first_left_partial_apply_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="**$>", replacement="|"
    )

    value_first_left_partial_apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="*$>", replacement="|"
    )

    value_first_left_partial_apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="$>", replacement="|"
    )

    function_first_left_partial_apply_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<$**", replacement="|"
    )

    function_first_left_partial_apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<$*", replacement="|"
    )

    function_first_left_partial_apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<$", replacement="|"
    )

    apply_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<|**", replacement="|"
    )

    apply_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<|*", replacement="|"
    )

    apply_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="<|", replacement="|"
    )

    compose_dict_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=".** ", replacement="| "
    )

    compose_tuple_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=".* ", replacement="| "
    )

    compose_op_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token=". ", replacement="| "
    )

    arg_placeholder_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_prefix,
        token="$",
        replacement="_",
    )

    placeholder_replacer = PlaceholderReplacer()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.binop_arg_nodes_to_skip: Set[int] = set()
        self.binop_nodes_to_eval: Set[int] = set()
        self.lexical_chain_stack: pyc.TraceStack = self.make_stack()
        with self.lexical_chain_stack.register_stack_state():
            self.cur_chain_placeholder_lambda: Optional[Callable[..., Any]] = None

    @pyc.register_handler(pyc.before_load_complex_symbol, reentrant=True)
    def handle_chain_placeholder_rewrites(
        self, ret, node: ast.expr, frame: FrameType, *_, **__
    ):
        with self.lexical_chain_stack.push():
            self.cur_chain_placeholder_lambda = None
        if not self.placeholder_replacer.search(
            node, allow_top_level=isinstance(node, ast.Call)
        ):
            return ret
        lambda_body = StatementMapper.augmentation_propagating_copy(node)
        assert isinstance(lambda_body, ast.expr)
        orig_ctr = self.placeholder_replacer.arg_ctr
        self.placeholder_replacer.rewrite(lambda_body, allow_top_level=True)
        ast_lambda = SingletonArgCounterMixin.create_placeholder_lambda(
            orig_ctr, lambda_body, frame.f_globals
        )
        ast_lambda.body = lambda_body
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        self.cur_chain_placeholder_lambda = evaluated_lambda
        return lambda: OptionalChainer.resolves_to_none_eventually

    @pyc.register_raw_handler(pyc.before_argument)
    def handle_before_arg(self, ret: object, *_, **__):
        if self.cur_chain_placeholder_lambda:
            return lambda: None
        else:
            return ret

    @pyc.register_raw_handler(pyc.after_load_complex_symbol)
    def handle_after_placeholder_chain(self, ret, *_, **__):
        override_ret = self.cur_chain_placeholder_lambda
        self.lexical_chain_stack.pop()
        if override_ret is None:
            return ret
        else:
            return override_ret

    @pyc.register_raw_handler((pyc.before_left_binop_arg, pyc.before_right_binop_arg))
    def maybe_skip_binop_arg(self, ret: object, node_id: int, *_, **__):
        if node_id in self.binop_arg_nodes_to_skip:
            self.binop_arg_nodes_to_skip.remove(node_id)
            return lambda: None
        else:
            return ret

    def transform_pipeline_placeholders(
        self,
        node: ast.expr,
        frame_globals: Dict[str, Any],
        full_node: Optional[ast.expr] = None,
    ) -> ast.Lambda:
        orig_ctr = self.placeholder_replacer.arg_ctr
        self.placeholder_replacer.rewrite(node, allow_top_level=False)
        return SingletonArgCounterMixin.create_placeholder_lambda(
            orig_ctr, full_node or node, frame_globals
        )

    @pyc.register_handler(
        pyc.before_right_binop_arg,
        when=pyc.Predicate(parent_is_bitor_op, static=True),
        reentrant=True,
    )
    def transform_pipeline_rhs_placeholders(
        self, ret: object, node: ast.expr, frame: FrameType, *_, **__
    ):
        if not self.get_augmentations(id(self.containing_ast_by_id.get(id(node)))):
            return ret
        if not self.placeholder_replacer.search(node, allow_top_level=False):
            return ret
        transformed = cast(
            ast.expr, StatementMapper.augmentation_propagating_copy(node)
        )
        ast_lambda = self.transform_pipeline_placeholders(transformed, frame.f_globals)
        ast_lambda.body = transformed
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        return lambda: evaluated_lambda

    def search_left_descendant_placeholder(self, node: ast.BinOp) -> int:
        num_traversals = 0
        while True:
            if not (
                self.get_augmentations(id(node))
                & {
                    self.pipeline_op_spec,
                    self.pipeline_tuple_op_spec,
                    self.pipeline_dict_op_spec,
                    self.pipeline_op_assign_spec,
                    self.value_first_left_partial_apply_op_spec,
                    self.value_first_left_partial_apply_tuple_op_spec,
                    self.value_first_left_partial_apply_dict_op_spec,
                }
            ):
                return -1
            node = node.left  # type: ignore[assignment]
            num_traversals += 1
            if (
                not isinstance(node, ast.BinOp)
                or not isinstance(node.op, ast.BitOr)
                or not self.get_augmentations(id(node))
            ):
                break
        if self.placeholder_replacer.search(node, allow_top_level=False):
            return num_traversals
        else:
            return -1

    @pyc.register_handler(
        pyc.before_binop,
        when=pyc.Predicate(lambda node: isinstance(node.op, ast.BitOr), static=True),
        reentrant=True,
    )
    def transform_pipeline_lhs_placeholders(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        num_left_traversals_to_lhs_placeholder_node = (
            self.search_left_descendant_placeholder(node)
        )
        if num_left_traversals_to_lhs_placeholder_node < 0:
            return ret
        self.binop_arg_nodes_to_skip.add(id(node.left))
        self.binop_arg_nodes_to_skip.add(id(node.right))
        self.binop_nodes_to_eval.add(id(node))
        transformed = cast(
            ast.BinOp, StatementMapper.augmentation_propagating_copy(node)
        )
        left_arg = transformed
        for _i in range(num_left_traversals_to_lhs_placeholder_node):
            left_arg = left_arg.left  # type: ignore[assignment]
        ast_lambda = self.transform_pipeline_placeholders(
            left_arg, frame.f_globals, full_node=transformed
        )
        ast_lambda.body = transformed
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        return lambda *_, **__: evaluated_lambda

    @pyc.register_handler(
        pyc.before_binop,
        when=pyc.Predicate(lambda node: isinstance(node.op, ast.BitOr), static=True),
        reentrant=True,
    )
    def transform_pipeline_op(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        if id(node) in self.binop_nodes_to_eval:
            self.binop_nodes_to_eval.remove(id(node))
            return ret
        this_node_augmentations = self.get_augmentations(id(node))
        if self.pipeline_op_spec in this_node_augmentations:
            return lambda x, y: y(x)
        elif self.pipeline_tuple_op_spec in this_node_augmentations:
            return lambda x, y: y(*x)
        elif self.pipeline_dict_op_spec in this_node_augmentations:
            return lambda x, y: y(**x)
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
        elif self.compose_dict_op_spec in this_node_augmentations:

            def __pipeline_dict_compose(f, g):
                def __tuple_composed(*args, **kwargs):
                    return f(**g(*args, **kwargs))

                return __tuple_composed

            return __pipeline_dict_compose
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
            return lambda x, y: (lambda *args, **kwargs: y(x, *args, **kwargs))
        elif (
            self.value_first_left_partial_apply_tuple_op_spec in this_node_augmentations
        ):
            return lambda x, y: (lambda *args, **kwargs: y(*x, *args, **kwargs))
        elif (
            self.value_first_left_partial_apply_dict_op_spec in this_node_augmentations
        ):
            return lambda x, y: (lambda *args, **kwargs: y(*args, **x, **kwargs))
        elif self.function_first_left_partial_apply_op_spec in this_node_augmentations:
            return lambda x, y: (lambda *args, **kwargs: x(y, *args, **kwargs))
        elif (
            self.function_first_left_partial_apply_tuple_op_spec
            in this_node_augmentations
        ):
            return lambda x, y: (lambda *args, **kwargs: x(*y, *args, **kwargs))
        elif (
            self.function_first_left_partial_apply_dict_op_spec
            in this_node_augmentations
        ):
            return lambda x, y: (lambda *args, **kwargs: x(*args, **y, **kwargs))
        elif self.apply_op_spec in this_node_augmentations:
            return lambda x, y: x(y)
        elif self.apply_tuple_op_spec in this_node_augmentations:
            return lambda x, y: x(*y)
        elif self.apply_dict_op_spec in this_node_augmentations:
            return lambda x, y: x(**y)
        else:
            return ret
