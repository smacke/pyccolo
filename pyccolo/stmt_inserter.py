# -*- coding: utf-8 -*-
import ast
import logging
from typing import cast, Callable, DefaultDict, Dict, List, Set, Union

from pyccolo import fast
from pyccolo.extra_builtins import (
    EMIT_EVENT,
    EXEC_SAVED_THUNK,
    TRACING_ENABLED,
    make_guard_name,
)
from pyccolo.trace_events import TraceEvent
from pyccolo.fast import EmitterMixin, make_test, make_composite_condition


logger = logging.getLogger(__name__)


_INSERT_STMT_TEMPLATE = '{}("{{evt}}", {{stmt_id}})'.format(EMIT_EVENT)


def _get_parsed_insert_stmt(stmt: ast.stmt, evt: TraceEvent) -> ast.Expr:
    with fast.location_of(stmt):
        return cast(
            ast.Expr,
            fast.parse(
                _INSERT_STMT_TEMPLATE.format(evt=evt.value, stmt_id=id(stmt))
            ).body[0],
        )


def _get_parsed_append_stmt(
    stmt: ast.stmt,
    ret_expr: ast.expr = None,
    evt: TraceEvent = TraceEvent.after_stmt,
    **kwargs,
) -> ast.Expr:
    with fast.location_of(stmt):
        ret = _get_parsed_insert_stmt(stmt, evt)
        if ret_expr is not None:
            kwargs["ret"] = ret_expr
        ret_value = cast(ast.Call, ret.value)
        ret_value.keywords = fast.kwargs(**kwargs)
    ret.lineno = getattr(stmt, "end_lineno", ret.lineno)
    return ret


class StripGlobalAndNonlocalDeclarations(ast.NodeTransformer):
    def visit_Global(self, node: ast.Global) -> ast.Pass:
        with fast.location_of(node):
            return fast.Pass()

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.Pass:
        with fast.location_of(node):
            return fast.Pass()


class StatementInserter(ast.NodeTransformer, EmitterMixin):
    def __init__(
        self,
        orig_to_copy_mapping: Dict[int, ast.AST],
        handler_predicate_by_event: DefaultDict[TraceEvent, Callable[..., bool]],
        guards: Set[str],
    ):
        EmitterMixin.__init__(
            self, orig_to_copy_mapping, handler_predicate_by_event, guards
        )
        self._global_nonlocal_stripper: StripGlobalAndNonlocalDeclarations = (
            StripGlobalAndNonlocalDeclarations()
        )

    def _handle_loop_body(
        self, node: Union[ast.For, ast.While], orig_body: List[ast.AST]
    ) -> List[ast.AST]:
        loop_node_copy = cast(
            Union[ast.For, ast.While], self.orig_to_copy_mapping[id(node)]
        )
        loop_node_copy = self._global_nonlocal_stripper.visit(loop_node_copy)
        loop_guard = make_guard_name(loop_node_copy)
        self.register_guard(loop_guard)
        with fast.location_of(loop_node_copy):
            if isinstance(node, ast.For):
                before_loop_evt = TraceEvent.before_for_loop_body
                after_loop_evt = TraceEvent.after_for_loop_iter
            else:
                before_loop_evt = TraceEvent.before_while_loop_body
                after_loop_evt = TraceEvent.after_while_loop_iter
            return [
                fast.If(
                    test=make_composite_condition(
                        [
                            make_test(TRACING_ENABLED),
                            make_test(loop_guard),
                            self.emit(
                                before_loop_evt, node, ret=fast.NameConstant(True)
                            )
                            if self.handler_predicate_by_event[before_loop_evt](node)
                            else None,
                        ]
                    ),
                    body=[
                        fast.Try(
                            body=orig_body,
                            handlers=[],
                            orelse=[],
                            finalbody=[
                                _get_parsed_append_stmt(
                                    cast(ast.stmt, loop_node_copy),
                                    evt=after_loop_evt,
                                    guard=fast.Str(loop_guard),
                                ),
                            ],
                        ),
                    ]
                    if self.handler_predicate_by_event[after_loop_evt](node)
                    else orig_body,
                    orelse=loop_node_copy.body,
                ),
            ]

    def _handle_function_body(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        orig_body: List[ast.AST],
    ) -> List[ast.AST]:
        fundef_copy = cast(
            Union[ast.FunctionDef, ast.AsyncFunctionDef],
            self.orig_to_copy_mapping[id(node)],
        )
        fundef_copy = self._global_nonlocal_stripper.visit(fundef_copy)
        function_guard = make_guard_name(fundef_copy)
        self.register_guard(function_guard)
        with fast.location_of(fundef_copy):
            return [
                fast.If(
                    test=make_composite_condition(
                        [
                            make_test(TRACING_ENABLED),
                            make_test(function_guard),
                            self.emit(
                                TraceEvent.before_function_body,
                                node,
                                ret=fast.NameConstant(True),
                            )
                            if self.handler_predicate_by_event[
                                TraceEvent.before_function_body
                            ](fundef_copy)
                            else None,
                        ]
                    ),
                    body=[
                        fast.Try(
                            body=orig_body,
                            handlers=[],
                            orelse=[],
                            finalbody=[
                                _get_parsed_append_stmt(
                                    cast(ast.stmt, fundef_copy),
                                    evt=TraceEvent.after_function_execution,
                                    guard=fast.Str(function_guard),
                                ),
                            ],
                        ),
                    ]
                    if self.handler_predicate_by_event[
                        TraceEvent.after_function_execution
                    ](fundef_copy)
                    else orig_body,
                    orelse=fundef_copy.body,
                ),
            ]

    def _make_main_and_after_stmt_stmts(
        self,
        outer_node: ast.AST,
        field_name: str,
        prev_stmt: ast.stmt,
        prev_stmt_copy: ast.stmt,
    ) -> List[ast.stmt]:
        if not self.handler_predicate_by_event[TraceEvent.after_stmt](prev_stmt_copy):
            return [self.visit(prev_stmt)]
        if (
            isinstance(prev_stmt, ast.Expr)
            and isinstance(outer_node, ast.Module)
            and field_name == "body"
        ):
            val = prev_stmt.value
            while isinstance(val, ast.Expr):
                val = val.value
            return [_get_parsed_append_stmt(prev_stmt_copy, ret_expr=val)]
        else:
            return [self.visit(prev_stmt)] + (
                []
                if isinstance(prev_stmt, ast.Return)
                else [_get_parsed_append_stmt(prev_stmt_copy)]
            )

    def _handle_stmt(
        self, node: ast.AST, field_name: str, inner_node: ast.stmt
    ) -> List[ast.stmt]:
        stmts_to_extend: List[ast.stmt] = []
        stmt_copy = cast(ast.stmt, self.orig_to_copy_mapping[id(inner_node)])
        main_and_maybe_after = self._make_main_and_after_stmt_stmts(
            node, field_name, inner_node, stmt_copy
        )
        if self.handler_predicate_by_event[TraceEvent.before_stmt](stmt_copy):
            with fast.location_of(stmt_copy):
                # TODO: save off the return value if the statement is an expression,
                #   and add a new statement that just evaluates to that returned value
                stmts_to_extend.append(
                    fast.If(
                        test=_get_parsed_insert_stmt(
                            stmt_copy, TraceEvent.before_stmt
                        ).value,
                        body=self._make_main_and_after_stmt_stmts(
                            node,
                            field_name,
                            fast.parse(f"{EXEC_SAVED_THUNK}()").body[0],
                            stmt_copy,
                        ),
                        orelse=main_and_maybe_after,
                    )
                )
        else:
            stmts_to_extend.extend(main_and_maybe_after)
        if (
            isinstance(node, ast.Module)
            and field_name == "body"
            and self.handler_predicate_by_event[TraceEvent.after_module_stmt](stmt_copy)
        ):
            assert not isinstance(inner_node, ast.Return)
            stmts_to_extend.append(
                _get_parsed_append_stmt(stmt_copy, evt=TraceEvent.after_module_stmt)
            )
        return stmts_to_extend

    def generic_visit(self, node):
        if self.is_tracing_disabled_context(node):
            return node
        for name, field in ast.iter_fields(node):
            if isinstance(field, ast.AST):
                setattr(node, name, self.visit(field))
            elif isinstance(field, list):
                new_field = []
                if isinstance(node, ast.Module) and name == "body":
                    if self.handler_predicate_by_event[TraceEvent.init_module](node):
                        with fast.location_of(node):
                            new_field.extend(
                                fast.parse(
                                    f'{EMIT_EVENT}("{TraceEvent.init_module.name}", '
                                    + f"{id(node)})"
                                ).body
                            )
                for inner_node in field:
                    if isinstance(inner_node, ast.stmt):
                        new_field.extend(self._handle_stmt(node, name, inner_node))
                    elif isinstance(inner_node, ast.AST):
                        new_field.append(self.visit(inner_node))
                    else:
                        new_field.append(inner_node)
                if name == "body":
                    if isinstance(node, ast.Module):
                        # TODO: should this go in try / finally to ensure it always gets executed?
                        if self.handler_predicate_by_event[TraceEvent.exit_module](
                            node
                        ):
                            with fast.location_of(new_field[-1]):
                                new_field.extend(
                                    fast.parse(
                                        f'{EMIT_EVENT}("{TraceEvent.exit_module.name}", '
                                        + f"{id(node)})"
                                    ).body
                                )
                    elif isinstance(node, (ast.For, ast.While)):
                        new_field = self._handle_loop_body(node, new_field)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        new_field = self._handle_function_body(node, new_field)
                setattr(node, name, new_field)
            else:
                continue
        return node
