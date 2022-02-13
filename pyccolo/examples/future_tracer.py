# -*- coding: utf-8 -*-
import ast
import builtins
import copy
import logging
import threading
import traceback
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Dict, List, Optional, Set, Tuple

import pyccolo as pyc
import pyccolo.fast as fast


try:
    from IPython import get_ipython
except ImportError:
    get_ipython = (lambda *_: None)


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

    def __call__(self, node: ast.AST) -> Tuple[ast.Expression, List[Future]]:
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


def _jump_to_non_internal_frame(tb):
    while tb is not None:
        pyccolo_seen = "pyccolo" in tb.tb_frame.f_code.co_filename
        concurrent_seen = "concurrent" in tb.tb_frame.f_code.co_filename
        if pyccolo_seen or concurrent_seen:
            tb = tb.tb_next
        else:
            break
    return tb


def _unwrap_exception(ex: Exception) -> Exception:
    tb = _jump_to_non_internal_frame(ex.__traceback__)
    if tb is None:
        return ex
    prev_tb_next = None
    while tb is not None and tb.tb_next is not None and tb.tb_next is not prev_tb_next:
        prev_tb_next = tb.tb_next
        tb.tb_next = _jump_to_non_internal_frame(prev_tb_next)
    return ex.with_traceback(tb)


class FutureTracer(pyc.BaseTracer):
    _MAX_WORKERS = 10
    _executor: List[Optional[ThreadPoolExecutor]] = [None]
    _async_variable_version_by_name: Dict[str, int] = Counter()
    _future_by_name_and_timestamp: Dict[Tuple[str, int], Future] = {}
    _waiters_by_future_id: Dict[int, Set[Future]] = defaultdict(set)
    _exec_counter_by_future_id: Dict[int, int] = {}
    _version_lock = threading.Lock()

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

    @pyc.load_name
    def handle_load_name(self, ret, node, *_, **__):
        if node.id in self._async_variable_version_by_name:
            try:
                return self._unwrap_future(ret)
            except Exception as ex:
                ex = _unwrap_exception(ex)
                relevant_cell = self._exec_counter_by_future_id.get(id(ret), None)
                if relevant_cell is not None:
                    logger.error("Exception occurred in cell %d:", relevant_cell)
                logger.error(
                    "".join(traceback.format_exception(type(ex), ex, ex.__traceback__))
                )
                return ret
        else:
            return ret

    @pyc.before_assign_rhs(when=pyc.BaseTracer.is_outer_stmt)
    def handle_assign_rhs(self, ret, node, frame, *_, **__):
        stmt = self.containing_stmt_by_id[id(node)]
        if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
            return ret
        unwrap_futures_expr, deps = self._future_unwrapper(node)
        unwrap_futures_code = compile(
            unwrap_futures_expr, "<pyccolo_file>", mode="eval"
        )
        async_var = stmt.targets[0].id
        with self._version_lock:
            self._async_variable_version_by_name[async_var] += 1
        current_version = self._async_variable_version_by_name[async_var]

        def work():
            old_fut = self._future_by_name_and_timestamp.get(
                (async_var, current_version - 1), None
            )
            for waiter in self._waiters_by_future_id.get(id(old_fut), []):
                # first, wait on everything that depends on the previous value to finish
                try:
                    self._unwrap_future(waiter)
                except:  # noqa: E722
                    pass
            # next, garbage collect the previous value
            self._waiters_by_future_id.pop(id(old_fut), None)
            self._future_by_name_and_timestamp.pop(
                (async_var, current_version - 1), None
            )
            retval = eval(unwrap_futures_code, frame.f_globals, frame.f_locals)
            with self._version_lock:
                if self._async_variable_version_by_name[async_var] == current_version:
                    # by using 'is_outer_stmt', we can be sure
                    # that setting the global is the right thing
                    frame.f_globals[async_var] = retval
            return retval

        ipy = get_ipython()
        current_cell = None if ipy is None else ipy.execution_count
        del ipy

        fut = self._executor[0].submit(work)
        if current_cell is not None:
            self._exec_counter_by_future_id[id(fut)] = current_cell
        for dep in deps:
            self._waiters_by_future_id[id(dep)].add(fut)
        self._future_by_name_and_timestamp[async_var, current_version] = fut
        return lambda: fut
