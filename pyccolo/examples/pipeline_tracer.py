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
import weakref
from contextlib import contextmanager
from types import FrameType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import pyccolo as pyc
import pyccolo.fast as fast
from pyccolo.examples.optional_chaining import OptionalChainer
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.trace_events import TraceEvent
from pyccolo.utils import clone_function

try:
    from executing.executing import find_node_ipython as orig_find_node_ipython

    orig_find_node_ipython_cloned = clone_function(orig_find_node_ipython)  # type: ignore[arg-type]
except ImportError:
    orig_find_node_ipython = None  # type: ignore[assignment]
    orig_find_node_ipython_cloned = None  # type: ignore[assignment]


_frame_to_node_mapping: "weakref.WeakValueDictionary[Tuple[str, int], ast.AST]" = (
    weakref.WeakValueDictionary()
)


def find_node_ipython(frame, last_i, stmts, source):
    decorator, node = orig_find_node_ipython_cloned(frame, last_i, stmts, source)
    if decorator is None and node is None:
        return None, _frame_to_node_mapping.get(
            (frame.f_code.co_filename, frame.f_lineno)
        )
    else:
        return decorator, node


def patch_find_node_ipython():
    if orig_find_node_ipython is None or orig_find_node_ipython_cloned is None:
        return
    orig_find_node_ipython.__code__ = find_node_ipython.__code__
    orig_find_node_ipython.__globals__["orig_find_node_ipython_cloned"] = (
        orig_find_node_ipython_cloned
    )
    orig_find_node_ipython.__globals__["_frame_to_node_mapping"] = (
        _frame_to_node_mapping
    )


class ExtractNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: Set[str] = set()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        before_names = set(self.names)
        self.generic_visit(node.body)
        for arg in fast.iter_arguments(node.args):
            if arg.arg not in before_names:
                self.names.discard(arg.arg)

    def visit_Name(self, node: ast.Name) -> None:
        if (
            node.id != "_"
            and id(node)
            not in PipelineTracer.augmented_node_ids_by_spec[
                PipelineTracer.arg_placeholder_spec
            ]
        ):
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
        cls,
        placeholder_names: List[str],
        orig_ctr: int,
        lambda_body: ast.expr,
        frame_globals: Dict[str, Any],
    ) -> ast.Lambda:
        num_lambda_args = cls._arg_ctr - orig_ctr
        lambda_args = []
        extra_defaults = ExtractNames.extract_names(lambda_body) - set(
            placeholder_names
        )
        for arg_idx in range(orig_ctr, orig_ctr + num_lambda_args):
            arg = f"_{arg_idx}"
            lambda_args.append(arg)
            extra_defaults.discard(arg)
        lambda_args.extend(placeholder_names)
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
    def __init__(self) -> None:
        self.mutate = False
        self.allow_top_level = False
        self.placeholder_names: Dict[str, None] = {}

    @contextmanager
    def disallow_top_level(self) -> Generator[None, None, None]:
        old_allow_top_level = self.allow_top_level
        try:
            self.allow_top_level = False
            yield
        finally:
            self.allow_top_level = old_allow_top_level

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.BinOp) or not PipelineTracer.get_augmentations(
            id(node.func)
        ):
            self.visit(node.func)
        if not self.allow_top_level:
            # defer visiting nested calls
            return
        with self.disallow_top_level():
            for arg in node.args:
                self.visit(arg)
            for kw in node.keywords:
                self.visit(kw.value)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        from pyccolo.examples.quick_lambda import QuickLambdaTracer

        if (
            isinstance(node.value, ast.Name)
            and node.value.id in QuickLambdaTracer.lambda_macros
        ):
            # defer visiting nested quick lambdas
            return
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
        if (
            id(node)
            not in PipelineTracer.augmented_node_ids_by_spec[
                PipelineTracer.arg_placeholder_spec
            ]
        ):
            return
        assert node.id.startswith("_")
        arg_ctr = self.arg_ctr
        if node.id == "_":
            self.arg_ctr += 1
        else:
            self.placeholder_names[node.id[1:]] = None
        if not self.mutate:
            return
        if node.id == "_":
            node.id = f"_{arg_ctr}"
        else:
            node.id = node.id[1:]
        PipelineTracer.augmented_node_ids_by_spec[
            PipelineTracer.arg_placeholder_spec
        ].discard(id(node))

    def search(
        self, node: Union[ast.AST, Sequence[ast.AST]], allow_top_level: bool
    ) -> bool:
        if isinstance(node, list):
            return any(
                self.search(inner, allow_top_level=allow_top_level) for inner in node
            )
        assert isinstance(node, ast.AST)
        orig_ctr = self.arg_ctr
        try:
            self.allow_top_level = allow_top_level
            self.visit(node)
            found = self.arg_ctr > orig_ctr or len(self.placeholder_names) > 0
        finally:
            self.arg_ctr = orig_ctr
            self.placeholder_names.clear()
        return found

    def rewrite(self, node: ast.expr, allow_top_level: bool) -> List[str]:
        old_mutate = self.mutate
        try:
            self.mutate = True
            self.allow_top_level = allow_top_level
            self.visit(node)
            ret = self.placeholder_names
        finally:
            self.mutate = old_mutate
            self.placeholder_names = {}
        return list(ret.keys())


def parent_is_bitor_op(node_or_id: Union[ast.expr, int]) -> bool:
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    parent = pyc.BaseTracer.containing_ast_by_id.get(node_id)
    return isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.BitOr)


@contextmanager
def allow_pipelines_in_loops_and_calls() -> Generator[None, None, None]:
    yield


def is_outer_or_allowlisted(node_or_id: Union[ast.AST, int]) -> bool:
    node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
    if pyc.is_outer_stmt(node_id):
        return True
    containing_stmt = pyc.BaseTracer.containing_stmt_by_id.get(node_id)
    parent_stmt = pyc.BaseTracer.parent_stmt_by_id.get(
        node_id if containing_stmt is None else id(containing_stmt)
    )
    while parent_stmt is not None:
        if isinstance(parent_stmt, ast.With):
            context_expr = parent_stmt.items[0].context_expr
            if (
                isinstance(context_expr, ast.Call)
                and isinstance(context_expr.func, ast.Name)
                and context_expr.func.id == allow_pipelines_in_loops_and_calls.__name__
            ):
                return True
        elif isinstance(parent_stmt, (ast.AsyncFunctionDef, ast.FunctionDef)):
            if any(
                isinstance(deco, ast.Name)
                and deco.id == allow_pipelines_in_loops_and_calls.__name__
                for deco in parent_stmt.decorator_list
            ):
                return True
        parent_stmt = pyc.BaseTracer.parent_stmt_by_id.get(id(parent_stmt))
    return False


class PipelineTracer(pyc.BaseTracer):

    global_guards_enabled = False

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
        self.placeholder_arg_position_cache: Dict[int, List[str]] = {}
        self.exc_to_propagate: Optional[Exception] = None
        with self.lexical_chain_stack.register_stack_state():
            self.cur_chain_placeholder_lambda: Optional[Callable[..., Any]] = None
        patch_find_node_ipython()
        allow_pipelines_name = allow_pipelines_in_loops_and_calls.__name__
        setattr(builtins, allow_pipelines_name, allow_pipelines_in_loops_and_calls)
        try:
            from IPython import get_ipython

            shell = get_ipython()
            if shell is not None:
                shell.user_ns[allow_pipelines_name] = allow_pipelines_in_loops_and_calls
        except ImportError:
            pass

    @pyc.register_handler(
        pyc.before_load_complex_symbol,
        when=is_outer_or_allowlisted,
        reentrant=True,
    )
    def handle_chain_placeholder_rewrites(
        self, ret, node: ast.expr, frame: FrameType, *_, **__
    ):
        with self.lexical_chain_stack.push():
            self.cur_chain_placeholder_lambda = None
        if not self.placeholder_replacer.search(node, allow_top_level=True):
            return ret
        __hide_pyccolo_frame__ = True
        _frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node
        node_copy = StatementMapper.bookkeeping_propagating_copy(node)
        assert isinstance(node_copy, ast.expr)
        orig_ctr = self.placeholder_replacer.arg_ctr
        lambda_body_parent_call = None
        lambda_body = node_copy
        while (
            isinstance(lambda_body, ast.Call)
            and isinstance(lambda_body.func, ast.Call)
            and not self.placeholder_replacer.search(
                cast(Sequence[ast.AST], lambda_body.args + lambda_body.keywords),
                allow_top_level=True,
            )
        ):
            lambda_body_parent_call = lambda_body
            lambda_body = lambda_body.func
        placeholder_names = self.placeholder_replacer.rewrite(
            lambda_body, allow_top_level=True
        )
        ast_lambda = SingletonArgCounterMixin.create_placeholder_lambda(
            placeholder_names, orig_ctr, lambda_body, frame.f_globals
        )
        ast_lambda.body = lambda_body
        if lambda_body_parent_call is None:
            node_to_eval: ast.expr = ast_lambda
        else:
            lambda_body_parent_call.func = ast_lambda
            node_to_eval = node_copy
        self.cur_chain_placeholder_lambda = lambda: __hide_pyccolo_frame__ and pyc.eval(
            node_to_eval, frame.f_globals, frame.f_locals
        )
        return lambda: OptionalChainer.resolves_to_none_eventually

    @pyc.register_raw_handler(pyc.before_argument, when=is_outer_or_allowlisted)
    def handle_before_arg(self, ret: object, *_, **__):
        if self.cur_chain_placeholder_lambda:
            return lambda: None
        else:
            return ret

    @pyc.register_raw_handler(
        pyc.after_load_complex_symbol, when=is_outer_or_allowlisted
    )
    def handle_after_placeholder_chain(self, ret, *_, **__):
        __hide_pyccolo_frame__ = True
        override_ret = self.cur_chain_placeholder_lambda
        self.lexical_chain_stack.pop()
        if override_ret is None:
            return ret
        try:
            return __hide_pyccolo_frame__ and override_ret()
        except Exception as e:
            self.exc_to_propagate = e
            raise e from None

    def should_propagate_handler_exception(
        self, _evt: TraceEvent, exc: Exception
    ) -> bool:
        if exc is self.exc_to_propagate:
            self.exc_to_propagate = None
            return True
        return False

    @pyc.register_raw_handler(
        (pyc.before_left_binop_arg, pyc.before_right_binop_arg),
        when=lambda node: parent_is_bitor_op(node) and is_outer_or_allowlisted(node),
    )
    def maybe_skip_binop_arg(self, ret: object, node_id: int, *_, **__):
        if node_id in self.binop_arg_nodes_to_skip:
            self.binop_arg_nodes_to_skip.remove(node_id)
            return lambda: None
        else:
            return ret

    def reorder_placeholder_names_for_prior_positions(
        self, node: ast.expr, placeholder_names: List[str]
    ) -> List[str]:
        if not isinstance(node, ast.BinOp):
            return placeholder_names
        prev_placeholders = self.placeholder_arg_position_cache.get(id(node))
        if prev_placeholders is None:
            return placeholder_names
        index_by_name = {
            name: (
                prev_placeholders.index(name)
                if name in prev_placeholders
                else float("inf")
            )
            for name in placeholder_names
        }
        return sorted(placeholder_names, key=lambda name: index_by_name[name])

    def transform_pipeline_placeholders(
        self,
        node: ast.expr,
        parent: ast.BinOp,
        frame_globals: Dict[str, Any],
        allow_top_level: bool,
        full_node: Optional[ast.expr] = None,
    ) -> ast.Lambda:
        orig_ctr = self.placeholder_replacer.arg_ctr
        placeholder_names = self.placeholder_replacer.rewrite(
            node, allow_top_level=allow_top_level
        )
        placeholder_names = self.reorder_placeholder_names_for_prior_positions(
            parent.left, placeholder_names
        )
        self.placeholder_arg_position_cache[id(parent)] = [
            name for name in placeholder_names if not name[1].isdigit()
        ]
        return SingletonArgCounterMixin.create_placeholder_lambda(
            placeholder_names, orig_ctr, full_node or node, frame_globals
        )

    @pyc.register_handler(
        pyc.before_right_binop_arg,
        when=lambda node: parent_is_bitor_op(node) and is_outer_or_allowlisted(node),
        reentrant=True,
    )
    def transform_pipeline_rhs_placeholders(
        self, ret: object, node: ast.expr, frame: FrameType, *_, **__
    ):
        parent: ast.BinOp = self.containing_ast_by_id.get(id(node))  # type: ignore[assignment]
        if not self.get_augmentations(id(parent)):
            return ret
        __hide_pyccolo_frame__ = True
        _frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node
        allow_top_level = not isinstance(node, ast.BinOp) or not self.get_augmentations(
            id(node)
        )
        if not self.placeholder_replacer.search(node, allow_top_level=allow_top_level):
            return ret
        transformed = cast(ast.expr, StatementMapper.bookkeeping_propagating_copy(node))
        ast_lambda = self.transform_pipeline_placeholders(
            transformed, parent, frame.f_globals, allow_top_level=allow_top_level
        )
        ast_lambda.body = transformed
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        return lambda: __hide_pyccolo_frame__ and evaluated_lambda

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
        when=lambda node: isinstance(node.op, ast.BitOr)
        and is_outer_or_allowlisted(node),
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
        __hide_pyccolo_frame__ = True
        _frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node
        self.binop_arg_nodes_to_skip.add(id(node.left))
        self.binop_arg_nodes_to_skip.add(id(node.right))
        self.binop_nodes_to_eval.add(id(node))
        transformed = cast(
            ast.BinOp, StatementMapper.bookkeeping_propagating_copy(node)
        )
        left_arg = transformed
        for _i in range(num_left_traversals_to_lhs_placeholder_node):
            left_arg = left_arg.left  # type: ignore[assignment]
        ast_lambda = self.transform_pipeline_placeholders(
            left_arg,
            node,
            frame.f_globals,
            allow_top_level=False,
            full_node=transformed,
        )
        ast_lambda.body = transformed
        evaluated_lambda = pyc.eval(ast_lambda, frame.f_globals, frame.f_locals)
        return lambda *_, **__: __hide_pyccolo_frame__ and evaluated_lambda

    @pyc.register_handler(
        pyc.before_binop,
        when=lambda node: isinstance(node.op, ast.BitOr)
        and is_outer_or_allowlisted(node),
        reentrant=True,
    )
    def transform_pipeline_op(
        self, ret: object, node: ast.BinOp, frame: FrameType, *_, **__
    ):
        if id(node) in self.binop_nodes_to_eval:
            self.binop_nodes_to_eval.remove(id(node))
            return ret
        __hide_pyccolo_frame__ = True
        _frame_to_node_mapping[frame.f_code.co_filename, frame.f_lineno] = node.left
        this_node_augmentations = self.get_augmentations(id(node))
        if self.pipeline_op_spec in this_node_augmentations:
            return lambda x, y: __hide_pyccolo_frame__ and y(x)
        elif self.pipeline_tuple_op_spec in this_node_augmentations:
            return lambda x, y: __hide_pyccolo_frame__ and y(*x)
        elif self.pipeline_dict_op_spec in this_node_augmentations:
            return lambda x, y: __hide_pyccolo_frame__ and y(**x)
        elif self.compose_op_spec in this_node_augmentations:

            def __pipeline_compose(f, g):
                def __composed(*args, **kwargs):
                    return __hide_pyccolo_frame__ and f(g(*args, **kwargs))

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
                return __hide_pyccolo_frame__ and val

            return lambda x, y: __hide_pyccolo_frame__ and assign_globals(x)
        elif self.value_first_left_partial_apply_op_spec in this_node_augmentations:
            return lambda x, y: __hide_pyccolo_frame__ and (
                lambda *args, **kwargs: __hide_pyccolo_frame__ and y(x, *args, **kwargs)
            )
        elif (
            self.value_first_left_partial_apply_tuple_op_spec in this_node_augmentations
        ):
            return lambda x, y: __hide_pyccolo_frame__ and (
                lambda *args, **kwargs: __hide_pyccolo_frame__
                and y(*x, *args, **kwargs)
            )
        elif (
            self.value_first_left_partial_apply_dict_op_spec in this_node_augmentations
        ):
            return lambda x, y: __hide_pyccolo_frame__ and (
                lambda *args, **kwargs: __hide_pyccolo_frame__
                and y(*args, **x, **kwargs)
            )
        elif self.function_first_left_partial_apply_op_spec in this_node_augmentations:
            return lambda x, y: __hide_pyccolo_frame__ and (
                lambda *args, **kwargs: __hide_pyccolo_frame__ and x(y, *args, **kwargs)
            )
        elif (
            self.function_first_left_partial_apply_tuple_op_spec
            in this_node_augmentations
        ):
            return lambda x, y: __hide_pyccolo_frame__ and (
                lambda *args, **kwargs: __hide_pyccolo_frame__
                and x(*y, *args, **kwargs)
            )
        elif (
            self.function_first_left_partial_apply_dict_op_spec
            in this_node_augmentations
        ):
            return lambda x, y: __hide_pyccolo_frame__ and (
                lambda *args, **kwargs: __hide_pyccolo_frame__
                and x(*args, **y, **kwargs)
            )
        elif self.apply_op_spec in this_node_augmentations:
            return lambda x, y: __hide_pyccolo_frame__ and x(y)
        elif self.apply_tuple_op_spec in this_node_augmentations:
            return lambda x, y: __hide_pyccolo_frame__ and x(*y)
        elif self.apply_dict_op_spec in this_node_augmentations:
            return lambda x, y: __hide_pyccolo_frame__ and x(**y)
        else:
            return ret
