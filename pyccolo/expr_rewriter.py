# -*- coding: utf-8 -*-
import ast
import logging
import sys
from contextlib import contextmanager
from typing import cast, Callable, DefaultDict, Dict, List, Optional, Set, Union

from pyccolo import fast
from pyccolo.extra_builtins import TRACING_ENABLED, make_guard_name
from pyccolo.fast import (
    EmitterMixin,
    make_test,
    make_composite_condition,
    subscript_to_slice,
)
from pyccolo.trace_events import TraceEvent


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class ExprRewriter(ast.NodeTransformer, EmitterMixin):
    def __init__(
        self,
        orig_to_copy_mapping: Dict[int, ast.AST],
        handler_predicate_by_event: DefaultDict[TraceEvent, Callable[..., bool]],
        guards: Set[str],
    ):
        EmitterMixin.__init__(
            self, orig_to_copy_mapping, handler_predicate_by_event, guards
        )
        self._top_level_node_for_symbol: Optional[ast.AST] = None

    def visit(self, node: ast.AST):
        ret = super().visit(node)
        if isinstance(node, ast.stmt):
            # We haven't inserted statements yet, and StatementInserter
            # needs the previous ids to be identical.
            assert ret is node
        return ret

    def visit_Name(self, node: ast.Name):
        if not isinstance(node.ctx, ast.Load):
            return node
        orig_node = node
        if self.handler_predicate_by_event[TraceEvent.load_name](orig_node):
            with fast.location_of(orig_node):
                node = self.emit(TraceEvent.load_name, orig_node, ret=node)  # type: ignore
        return node

    @contextmanager
    def attrsub_context(self, top_level_node: Optional[ast.AST]):
        old = self._top_level_node_for_symbol
        if old is None or top_level_node is None:
            # entering context when we are already inside chain is a no-op,
            # but we can specify a context of not being in chain if we are
            # inside one (in order to support arguments)
            self._top_level_node_for_symbol = top_level_node
        yield
        self._top_level_node_for_symbol = old

    @property
    def _inside_attrsub_load_chain(self):
        return self._top_level_node_for_symbol is not None

    @staticmethod
    def _get_attrsub_event(node: Union[ast.Attribute, ast.Subscript]) -> TraceEvent:
        is_subscript = isinstance(node, ast.Subscript)
        if isinstance(node.ctx, ast.Load):
            return (
                TraceEvent.before_subscript_load
                if is_subscript
                else TraceEvent.before_attribute_load
            )
        elif isinstance(node.ctx, ast.Store):
            return (
                TraceEvent.before_subscript_store
                if is_subscript
                else TraceEvent.before_attribute_store
            )
        elif isinstance(node.ctx, ast.Del):
            return (
                TraceEvent.before_subscript_del
                if is_subscript
                else TraceEvent.before_attribute_del
            )
        else:
            raise ValueError("unknown context: %s", node.ctx)

    def visit_Attribute(self, node: ast.Attribute, call_context=False):
        with fast.location_of(node.value):
            attr_node = cast(ast.Attribute, node)
            attr_or_sub = fast.Str(attr_node.attr)
        return self.visit_Attribute_or_Subscript(
            node, attr_or_sub, self._get_attrsub_event(node), call_context=call_context
        )

    def _maybe_convert_ast_subscript(self, subscript: ast.AST) -> ast.expr:
        if isinstance(subscript, ast.Index):
            return self.visit(subscript.value)  # type: ignore
        elif isinstance(subscript, ast.Slice):
            lower = (
                fast.NameConstant(None)
                if subscript.lower is None
                else self.visit(subscript.lower)
            )
            upper = (
                fast.NameConstant(None)
                if subscript.upper is None
                else self.visit(subscript.upper)
            )
            return fast.Call(
                func=fast.Name("slice", ast.Load()),
                args=[lower, upper]
                + ([] if subscript.step is None else [self.visit(subscript.step)]),
                keywords=[],
            )
        elif isinstance(subscript, (ast.ExtSlice, ast.Tuple)):
            return cast(ast.expr, subscript)
        else:
            return self.visit(subscript)

    def visit_Subscript(self, node: ast.Subscript, call_context=False):
        evt_to_use = self._get_attrsub_event(node)
        with self.attrsub_context(None):
            with fast.location_of(
                node.slice if hasattr(node.slice, "lineno") else node.value
            ):
                slc = self._maybe_convert_ast_subscript(node.slice)
                if isinstance(slc, (ast.ExtSlice, ast.Tuple)):
                    elts = slc.elts if isinstance(slc, ast.Tuple) else slc.dims  # type: ignore
                    elts = [self._maybe_convert_ast_subscript(elt) for elt in elts]  # type: ignore
                    slc = fast.Tuple(elts, ast.Load())
                if self.handler_predicate_by_event[TraceEvent.before_subscript_slice](
                    node
                ):
                    slc = self.emit(TraceEvent.before_subscript_slice, node, ret=slc)
                if self.handler_predicate_by_event[TraceEvent.after_subscript_slice](
                    node
                ):
                    slc = self.emit(TraceEvent.after_subscript_slice, node, ret=slc)
                if self.handler_predicate_by_event[evt_to_use](node.slice):
                    replacement_slice: ast.expr = self.emit(
                        TraceEvent._load_saved_slice, node.slice
                    )
                else:
                    replacement_slice = slc
                if sys.version_info >= (3, 9):
                    node.slice = replacement_slice
                else:
                    node.slice = fast.Index(replacement_slice)
        return self.visit_Attribute_or_Subscript(
            node, slc, evt_to_use, call_context=call_context
        )

    def _maybe_wrap_symbol_in_before_after_tracing(
        self,
        node,
        call_context=False,
        orig_node_id=None,
    ):
        if self._inside_attrsub_load_chain:
            return node
        orig_node = node
        orig_node_id = orig_node_id or id(orig_node)

        ctx = getattr(orig_node, "ctx", ast.Load())
        is_load = isinstance(ctx, ast.Load)
        if not is_load:
            return node

        with fast.location_of(node):
            extra_kwargs = dict(call_context=fast.NameConstant(call_context))
            if self.handler_predicate_by_event[TraceEvent.before_load_complex_symbol](
                orig_node
            ):
                node = self.emit(
                    TraceEvent.before_load_complex_symbol,
                    orig_node_id,
                    ret=node,
                    **extra_kwargs,
                )
            if self.handler_predicate_by_event[TraceEvent.after_load_complex_symbol](
                orig_node
            ):
                node = self.emit(
                    TraceEvent.after_load_complex_symbol,
                    orig_node_id,
                    ret=node,
                    **extra_kwargs,
                )
        # end location_of(node)
        return node

    def visit_Attribute_or_Subscript(
        self,
        node: Union[ast.Attribute, ast.Subscript],
        attr_or_sub: ast.expr,
        evt_to_use: TraceEvent,
        call_context: bool = False,
    ):
        orig_node = node
        orig_node_id = id(orig_node)
        with fast.location_of(node.value):
            is_subscript = isinstance(node, ast.Subscript)
            should_emit_evt = self.handler_predicate_by_event[evt_to_use](orig_node)
            with self.attrsub_context(node):
                extra_keywords: Dict[str, ast.AST] = {}
                if isinstance(node.value, ast.Name):
                    extra_keywords["obj_name"] = fast.Str(node.value.id)
                node.value = self.visit(node.value)
                if should_emit_evt:
                    if is_subscript:
                        subscript_name = None
                        if isinstance(node, ast.Subscript):
                            slice_val = subscript_to_slice(node)
                            if isinstance(slice_val, ast.Name):
                                # TODO: this should be more general than
                                #  just simple ast.Name subscripts
                                subscript_name = slice_val.id
                        extra_keywords["subscript_name"] = (
                            fast.NameConstant(None)
                            if subscript_name is None
                            else fast.Str(subscript_name)
                        )
                    node.value = self.emit(
                        evt_to_use,
                        orig_node_id,
                        ret=node.value,
                        attr_or_subscript=attr_or_sub,
                        call_context=fast.NameConstant(call_context),
                        top_level_node_id=self.get_copy_id_ast(
                            self._top_level_node_for_symbol
                        ),
                        **extra_keywords,
                    )
        # end fast.location_of(node.value)
        if isinstance(node.ctx, ast.Load):
            after_evt = (
                TraceEvent.after_subscript_load
                if is_subscript
                else TraceEvent.after_attribute_load
            )
            if self.handler_predicate_by_event[after_evt](orig_node):
                with fast.location_of(node):
                    node = self.emit(  # type: ignore
                        after_evt,
                        orig_node_id,
                        ret=node,
                        call_context=fast.NameConstant(call_context),
                    )

        return self._maybe_wrap_symbol_in_before_after_tracing(
            node, orig_node_id=orig_node_id
        )

    def _get_replacement_args(self, args, keywords: bool):
        replacement_args = []
        for arg in args:
            is_starred = isinstance(arg, ast.Starred)
            is_kwstarred = keywords and arg.arg is None
            if keywords or is_starred:
                maybe_kwarg = getattr(arg, "value")
            else:
                maybe_kwarg = arg
            with fast.location_of(maybe_kwarg):
                with self.attrsub_context(None):
                    new_arg_value = self.visit(maybe_kwarg)
                if self.handler_predicate_by_event[TraceEvent.argument](maybe_kwarg):
                    with self.attrsub_context(None):
                        new_arg_value = cast(
                            ast.expr,
                            self.emit(
                                TraceEvent.argument,
                                maybe_kwarg,
                                ret=new_arg_value,
                                is_starred=fast.NameConstant(is_starred),
                                is_kwstarred=fast.NameConstant(is_kwstarred),
                            ),
                        )
                if keywords or is_starred:
                    setattr(arg, "value", new_arg_value)
                else:
                    arg = new_arg_value
            replacement_args.append(arg)
        return replacement_args

    def visit_Call(self, node: ast.Call):
        orig_node = node
        orig_node_id = id(orig_node)

        with self.attrsub_context(node):
            if isinstance(node.func, ast.Attribute):
                node.func = self.visit_Attribute(node.func, call_context=True)
            elif isinstance(node.func, ast.Subscript):
                node.func = self.visit_Subscript(node.func, call_context=True)
            else:
                node.func = self.visit(node.func)

        # TODO: need a way to rewrite ast of subscript args,
        #  and to process these separately from outer rewrite

        node.args = self._get_replacement_args(node.args, False)
        node.keywords = self._get_replacement_args(node.keywords, True)

        # in order to ensure that the args are processed with appropriate active scope,
        # we need to make sure not to use the active namespace scope on args (in the case
        # of a function call on an ast.Attribute).
        #
        # We do so by emitting an "enter argument list", whose handler pushes the current active
        # scope while we process each argument. The "end argument list" event will then restore
        # the active scope.
        #
        # This effectively rewrites function calls as follows:
        # f(a, b, ..., c) -> trace(f, 'enter argument list')(a, b, ..., c)
        if self.handler_predicate_by_event[TraceEvent.before_call](orig_node):
            with fast.location_of(node.func):
                node.func = self.emit(
                    TraceEvent.before_call,
                    orig_node_id,
                    ret=node.func,
                    call_node_id=self.get_copy_id_ast(orig_node_id),
                )

        # f(a, b, ..., c) -> trace(f(a, b, ..., c), 'exit argument list')
        if self.handler_predicate_by_event[TraceEvent.after_call](orig_node):
            with fast.location_of(node):
                node = self.emit(
                    TraceEvent.after_call,
                    orig_node_id,
                    ret=node,
                    call_node_id=self.get_copy_id_ast(orig_node_id),
                )

        return self._maybe_wrap_symbol_in_before_after_tracing(
            node, call_context=True, orig_node_id=orig_node_id
        )

    def visit_Expr(self, node: ast.Expr) -> ast.Expr:
        with fast.location_of(node):
            node.value = self.visit(node.value)
            if self.handler_predicate_by_event[TraceEvent.after_expr_stmt](node):
                node.value = self.emit(TraceEvent.after_expr_stmt, node, ret=node.value)
            return node

    def visit_With(self, node: ast.With) -> ast.With:
        if self.is_tracing_disabled_context(node):
            return node
        else:
            return cast(ast.With, self.generic_visit(node))

    def visit_Assign(self, node: ast.Assign):
        return self.visit_Assign_or_AugAssign(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        return self.visit_Assign_or_AugAssign(node)

    def visit_Assign_or_AugAssign(self, node: Union[ast.Assign, ast.AugAssign]):
        if isinstance(node, ast.Assign):
            new_targets = []
            for target in node.targets:
                new_targets.append(self.visit(target))
            node.targets = new_targets
        else:
            node.target = self.visit(node.target)
        orig_value = node.value
        orig_value_id = id(orig_value)
        if isinstance(node, ast.Assign):
            before_evt = TraceEvent.before_assign_rhs
            after_evt = TraceEvent.after_assign_rhs
        else:
            before_evt = TraceEvent.before_augassign_rhs
            after_evt = TraceEvent.after_augassign_rhs
        with fast.location_of(node.value):
            node.value = self.visit(node.value)
            if self.handler_predicate_by_event[before_evt](orig_value):
                node.value = self.emit(before_evt, orig_value_id, ret=node.value)
            if self.handler_predicate_by_event[after_evt](orig_value):
                node.value = self.emit(after_evt, orig_value_id, ret=node.value)
        return node

    def visit_If(self, node: ast.If):
        with fast.location_of(node):
            node.test = self.visit(node.test)
            node.body = [self.visit(stmt) for stmt in node.body]
            node.orelse = [self.visit(stmt) for stmt in node.orelse]
            if self.handler_predicate_by_event[TraceEvent.after_if_test](node):
                node.test = self.emit(TraceEvent.after_if_test, node, ret=node.test)
            return node

    def visit_Lambda(self, node: ast.Lambda):
        assert isinstance(getattr(node, "ctx", ast.Load()), ast.Load)
        untraced_lam = cast(ast.Lambda, self.orig_to_copy_mapping[id(node)])
        ret_node: ast.Lambda = cast(ast.Lambda, self.generic_visit(node))
        with fast.location_of(node):
            ret_node.body = fast.IfExp(
                test=make_composite_condition(
                    [
                        make_test(TRACING_ENABLED),
                        self.emit(
                            TraceEvent.before_lambda_body,
                            node,
                            ret=fast.NameConstant(True),
                        )
                        if self.handler_predicate_by_event[
                            TraceEvent.before_lambda_body
                        ](untraced_lam)
                        else None,
                    ]
                ),
                body=ret_node.body,
                orelse=untraced_lam.body,
            )
            if self.handler_predicate_by_event[TraceEvent.after_lambda_body](
                untraced_lam
            ):
                ret_node.body = self.emit(
                    TraceEvent.after_lambda_body, node, ret=ret_node.body
                )
            if self.handler_predicate_by_event[TraceEvent.before_lambda](untraced_lam):
                ret_node = self.emit(TraceEvent.before_lambda, node, ret=ret_node)  # type: ignore
            if self.handler_predicate_by_event[TraceEvent.after_lambda](untraced_lam):
                ret_node = self.emit(TraceEvent.after_lambda, node, ret=ret_node)  # type: ignore
        return ret_node

    def visit_While(self, node: ast.While):
        for name, field in ast.iter_fields(node):
            if name == "test":
                loop_node_copy = cast(ast.While, self.orig_to_copy_mapping[id(node)])
                loop_guard = make_guard_name(loop_node_copy)
                self.register_guard(loop_guard)
                with fast.location_of(node):
                    visited_test = self.visit(field)
                    if self.handler_predicate_by_event[TraceEvent.after_while_test](
                        node
                    ):
                        visited_test = self.emit(
                            TraceEvent.after_while_test, node, ret=visited_test
                        )
                    node.test = fast.IfExp(
                        test=make_composite_condition(
                            [
                                make_test(TRACING_ENABLED),
                                make_test(loop_guard),
                            ]
                        ),
                        body=visited_test,
                        orelse=loop_node_copy.test,
                    )
            elif isinstance(field, list):
                setattr(node, name, [self.visit(elt) for elt in field])
            else:
                setattr(node, name, self.visit(field))
        return node

    @staticmethod
    def _ast_container_to_literal_trace_evt(
        node: Union[ast.Dict, ast.List, ast.Set, ast.Tuple], before: bool
    ) -> TraceEvent:
        if isinstance(node, ast.Dict):
            return (
                TraceEvent.before_dict_literal
                if before
                else TraceEvent.after_dict_literal
            )
        elif isinstance(node, ast.List):
            return (
                TraceEvent.before_list_literal
                if before
                else TraceEvent.after_list_literal
            )
        elif isinstance(node, ast.Set):
            return (
                TraceEvent.before_set_literal
                if before
                else TraceEvent.after_set_literal
            )
        elif isinstance(node, ast.Tuple):
            return (
                TraceEvent.before_tuple_literal
                if before
                else TraceEvent.after_tuple_literal
            )
        else:
            raise TypeError("invalid ast node: %s", ast.dump(node))

    def visit_literal(
        self,
        node: Union[ast.Dict, ast.List, ast.Set, ast.Tuple],
        should_inner_visit=True,
    ):
        untraced_lit = self.orig_to_copy_mapping[id(node)]
        ret_node: ast.expr = node
        if should_inner_visit:
            ret_node = cast(ast.expr, self.generic_visit(node))
        if not isinstance(getattr(node, "ctx", ast.Load()), ast.Load):
            return ret_node
        with fast.location_of(node):
            lit_before_evt = self._ast_container_to_literal_trace_evt(node, before=True)
            if self.handler_predicate_by_event[lit_before_evt](untraced_lit):
                ret_node = self.emit(lit_before_evt, node, ret=ret_node)
            lit_after_evt = self._ast_container_to_literal_trace_evt(node, before=False)
            if self.handler_predicate_by_event[lit_after_evt](untraced_lit):
                ret_node = self.emit(lit_after_evt, node, ret=ret_node)
        return ret_node

    def visit_List(self, node: ast.List):
        return self.visit_List_or_Set_or_Tuple(node)

    def visit_Set(self, node: ast.Set):
        return self.visit_List_or_Set_or_Tuple(node)

    def visit_Tuple(self, node: ast.Tuple):
        return self.visit_List_or_Set_or_Tuple(node)

    @staticmethod
    def _ast_container_to_elt_trace_evt(
        node: Union[ast.List, ast.Set, ast.Tuple]
    ) -> TraceEvent:
        if isinstance(node, ast.List):
            return TraceEvent.list_elt
        elif isinstance(node, ast.Set):
            return TraceEvent.set_elt
        elif isinstance(node, ast.Tuple):
            return TraceEvent.tuple_elt
        else:
            raise TypeError("invalid ast node: %s", ast.dump(node))

    def visit_List_or_Set_or_Tuple(self, node: Union[ast.List, ast.Set, ast.Tuple]):
        traced_elts: List[ast.expr] = []
        is_load = isinstance(getattr(node, "ctx", ast.Load()), ast.Load)
        saw_starred = False
        elt_trace_evt = self._ast_container_to_elt_trace_evt(node)
        for i, elt in enumerate(node.elts):
            if isinstance(elt, ast.Starred):
                saw_starred = True
                traced_elts.append(self.visit(elt))
                continue
            elif not is_load or not self.handler_predicate_by_event[elt_trace_evt](
                node
            ):
                traced_elts.append(self.visit(elt))
                continue
            with fast.location_of(elt):
                traced_elts.append(
                    self.emit(
                        elt_trace_evt,
                        elt,
                        ret=self.visit(elt),
                        index=fast.NameConstant(None) if saw_starred else fast.Num(i),
                        container_node_id=self.get_copy_id_ast(node),
                    )
                )
        node.elts = traced_elts
        return self.visit_literal(node, should_inner_visit=False)

    def visit_Dict(self, node: ast.Dict):
        traced_keys: List[Optional[ast.expr]] = []
        traced_values: List[ast.expr] = []
        for k, v in zip(node.keys, node.values):
            is_dict_unpack = k is None
            if is_dict_unpack:
                traced_keys.append(None)
            else:
                with fast.location_of(k):
                    traced_key = self.visit(k)
                    if self.handler_predicate_by_event[TraceEvent.dict_key](k):
                        traced_key = self.emit(
                            TraceEvent.dict_key,
                            k,
                            ret=traced_key,
                            value_node_id=self.get_copy_id_ast(v),
                            dict_node_id=self.get_copy_id_ast(node),
                        )
                    traced_keys.append(traced_key)
            with fast.location_of(v):
                if is_dict_unpack:
                    key_node_id_ast: ast.AST = fast.NameConstant(None)
                else:
                    key_node_id_ast = self.get_copy_id_ast(k)
                traced_value = self.visit(v)
                if self.handler_predicate_by_event[TraceEvent.dict_value](v):
                    traced_value = self.emit(
                        TraceEvent.dict_value,
                        v,
                        ret=traced_value,
                        key_node_id=key_node_id_ast,
                        dict_node_id=self.get_copy_id_ast(node),
                    )
                traced_values.append(traced_value)
        node.keys = traced_keys
        node.values = traced_values
        return self.visit_literal(node, should_inner_visit=False)

    def visit_Return(self, node: ast.Return):
        if node.value is None:
            return node
        orig_node_value = node.value
        with fast.location_of(node):
            node.value = self.visit(node.value)
            if self.handler_predicate_by_event[TraceEvent.before_return](
                orig_node_value
            ):
                node.value = self.emit(
                    TraceEvent.before_return, orig_node_value, ret=node.value
                )
            if self.handler_predicate_by_event[TraceEvent.after_return](
                orig_node_value
            ):
                node.value = self.emit(
                    TraceEvent.after_return, orig_node_value, ret=node.value
                )
        return node

    def visit_Delete(self, node: ast.Delete):
        ret = cast(ast.Delete, self.generic_visit(node))
        for target in ret.targets:
            target.ctx = ast.Del()  # type: ignore
        return ret

    def visit_BinOp(self, node: ast.BinOp) -> Union[ast.BinOp, ast.Call]:
        untraced_node = self.orig_to_copy_mapping[id(node)]
        op = node.op

        for attr, operand_evt in [
            ("left", TraceEvent.left_binop_arg),
            ("right", TraceEvent.right_binop_arg),
        ]:
            operand_node = getattr(node, attr)
            if self.handler_predicate_by_event[operand_evt](operand_node):
                with fast.location_of(operand_node):
                    setattr(
                        node,
                        attr,
                        self.emit(
                            operand_evt, operand_node, ret=self.visit(operand_node)
                        ),
                    )
            else:
                setattr(node, attr, self.visit(operand_node))

        ret: Union[ast.BinOp, ast.Call] = node
        if self.handler_predicate_by_event[TraceEvent.before_binop](untraced_node):
            with fast.location_of(node):
                ret = self.emit(
                    TraceEvent.before_binop,
                    node,
                    ret=self.make_lambda(
                        body=fast.BinOp(
                            op=op,
                            left=fast.Name(id="x", ctx=ast.Load()),
                            right=fast.Name(id="y", ctx=ast.Load()),
                        ),
                        args=[fast.arg("x", None), fast.arg("y", None)],
                    ),
                    before_expr_args=[node.left, node.right],
                )
        if self.handler_predicate_by_event[TraceEvent.after_binop](untraced_node):
            with fast.location_of(node):
                ret = self.emit(TraceEvent.after_binop, node, ret=ret)
        return ret

    def visit_Compare(self, node: ast.Compare) -> Union[ast.Compare, ast.Call]:
        # TODO: this is pretty similar to BinOp above; maybe can dedup some code
        untraced_node = self.orig_to_copy_mapping[id(node)]

        if self.handler_predicate_by_event[TraceEvent.left_compare_arg](node.left):
            with fast.location_of(node.left):
                node.left = self.emit(
                    TraceEvent.left_compare_arg, node.left, ret=self.visit(node.left)
                )
        else:
            node.left = self.visit(node.left)

        for idx, comparator in enumerate(node.comparators):
            if self.handler_predicate_by_event[TraceEvent.compare_arg](comparator):
                with fast.location_of(comparator):
                    node.comparators[idx] = self.emit(
                        TraceEvent.compare_arg, comparator, ret=self.visit(comparator)
                    )
            else:
                node.comparators[idx] = self.visit(comparator)

        ret: Union[ast.Compare, ast.Call] = node
        if self.handler_predicate_by_event[TraceEvent.before_compare](untraced_node):
            with fast.location_of(node):
                ret = self.emit(
                    TraceEvent.before_compare,
                    node,
                    ret=self.make_lambda(
                        body=fast.Compare(
                            ops=node.ops,
                            left=fast.Name(id="x", ctx=ast.Load()),
                            comparators=[
                                fast.Name(id=f"y_{i}", ctx=ast.Load())
                                for i in range(len(node.comparators))
                            ],
                        ),
                        args=[fast.arg("x", None)]
                        + [
                            fast.arg(f"y_{i}", None)
                            for i in range(len(node.comparators))
                        ],
                    ),
                    before_expr_args=[node.left] + node.comparators,
                )
        if self.handler_predicate_by_event[TraceEvent.after_compare](untraced_node):
            with fast.location_of(node):
                ret = self.emit(TraceEvent.after_compare, node, ret=ret)
        return ret

    if sys.version_info < (3, 8):

        def visit_Ellipsis(self, node: ast.Ellipsis):
            if self.handler_predicate_by_event[TraceEvent.ellipses](node):
                with fast.location_of(node):
                    return self.emit(TraceEvent.ellipses, node, ret=node)
            else:
                return node

    else:

        def visit_Constant(self, node: ast.Constant):
            if node.value is ... and self.handler_predicate_by_event[
                TraceEvent.ellipses
            ](node):
                with fast.location_of(node):
                    return self.emit(TraceEvent.ellipses, node, ret=node)
            else:
                return node
