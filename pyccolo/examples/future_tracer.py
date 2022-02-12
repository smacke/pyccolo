# -*- coding: utf-8 -*-
import ast
import builtins
import copy
import logging
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import pyccolo as pyc
import pyccolo.fast as fast


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


_UNWRAP_FUTURE_EXTRA_BUILTIN = "_Xix_PYCCOLO_UNWRAP_FUTURE"
_FUT_TAB_EXTRA_BUILTIN = "_Xix_PYCCOLO_FUTURE_TABLE"


class FutureUnwrapper(ast.NodeTransformer):
    def __init__(
        self,
        async_vars: Dict[str, int],
        future_by_name_and_timestamp: Dict[Tuple[str, int], Future],
    ) -> None:
        self._async_vars = async_vars
        self._future_by_name_and_timestamp = future_by_name_and_timestamp
        self._deps: List[Future] = []

    def __call__(self, node: ast.AST) -> Tuple[ast.AST, List[Future]]:
        transformed_node = self.visit(copy.deepcopy(node))
        deps, self._deps = self._deps, []
        return ast.Expression(transformed_node), deps

    def visit_Name(self, node: ast.Name):
        assert isinstance(node.ctx, ast.Load)
        current_version = self._async_vars.get(node.id, None)
        if current_version is None:
            return node
        else:
            self._deps.append(
                self._future_by_name_and_timestamp[node.id, current_version]
            )
            with fast.location_of(node):
                return fast.Call(
                    func=fast.Name(_UNWRAP_FUTURE_EXTRA_BUILTIN, ast.Load()),
                    args=[
                        fast.Subscript(
                            value=fast.Name(_FUT_TAB_EXTRA_BUILTIN, ast.Load()),
                            slice=fast.Tuple(
                                elts=[fast.Str(node.id), fast.Num(current_version)],
                                ctx=ast.Load(),
                            ),
                            ctx=ast.Load(),
                        )
                    ],
                )


class FutureTracer(pyc.BaseTracer):
    _MAX_WORKERS = 10
    _executor: List[Optional[ThreadPoolExecutor]] = [None]
    _async_variable_version_by_name: Dict[str, int] = Counter()
    _future_by_name_and_timestamp: Dict[Tuple[str, int], Future] = {}
    _waiters_by_future_id: Dict[int, Set[Future]] = defaultdict(set)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setattr(builtins, _UNWRAP_FUTURE_EXTRA_BUILTIN, self._unwrap_future)
        setattr(builtins, _FUT_TAB_EXTRA_BUILTIN, self._future_by_name_and_timestamp)
        self._future_unwrapper = FutureUnwrapper(
            self._async_variable_version_by_name, self._future_by_name_and_timestamp
        )
        if self._executor[0] is None:
            self._executor[0] = ThreadPoolExecutor(max_workers=self._MAX_WORKERS)

    @classmethod
    def clear_instance(cls):
        # TODO: replace all created futures with their materialized
        #  values in the stack frames that reference them
        if cls._executor[0] is not None:
            cls._executor[0].shutdown()
            cls._executor[0] = None
        super().clear_instance()

    def _unwrap_future(self, fut):
        if isinstance(fut, Future):
            return fut.result()
        else:
            return fut

    @pyc.load_name(when=pyc.BaseTracer.is_outer_stmt, reentrant=True)
    def handle_load_name(self, ret, node, *_, **__):
        if node.id in self._async_variable_version_by_name:
            return self._unwrap_future(ret)
        else:
            return ret

    @pyc.before_assign_rhs(when=pyc.BaseTracer.is_outer_stmt)
    def handle_assign_rhs(self, ret, node, frame, *_, **__):
        stmt = self.containing_stmt_by_id[id(node)]
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return ret
        unwrap_futures_expr, deps = self._future_unwrapper(node)
        unwrap_futures_code = compile(unwrap_futures_expr, "<file>", mode="eval")
        async_var = stmt.targets[0].id
        self._async_variable_version_by_name[async_var] += 1
        current_version = self._async_variable_version_by_name[async_var]

        def work():
            old_fut = self._future_by_name_and_timestamp.get(
                (async_var, current_version - 1), None
            )
            for waiter in self._waiters_by_future_id.get(id(old_fut), []):
                # first, wait on everything that depends on the previous value to finish
                self._unwrap_future(waiter)
            # next, garbage collect the previous value
            self._waiters_by_future_id.pop(id(old_fut), None)
            self._future_by_name_and_timestamp.pop(
                (async_var, current_version - 1), None
            )
            return eval(unwrap_futures_code, frame.f_globals, frame.f_locals)

        fut = self._executor[0].submit(work)
        for dep in deps:
            self._waiters_by_future_id[id(dep)].add(fut)
        self._future_by_name_and_timestamp[async_var, current_version] = fut
        return lambda: fut
