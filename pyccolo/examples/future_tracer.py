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

    def get_ipython(*_):
        return None


try:
    from nbsafety.singletons import nbs
except ImportError:

    def nbs(*_):
        return None


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
        self._async_var: Optional[str] = None

    def __call__(
        self, node: ast.AST, async_var: str
    ) -> Tuple[ast.Expression, List[Future]]:
        self._async_var = async_var
        transformed_node = self.visit(copy.deepcopy(node))
        deps, self._deps = self._deps, []
        return ast.Expression(transformed_node), deps

    def visit_Name(self, node: ast.Name):
        assert isinstance(node.ctx, ast.Load)
        current_version = self._async_vars.get(node.id, None)
        if current_version is None:
            return node
        else:
            if node.id != self._async_var:
                # exclude usages of same var since we don't want to wait on self
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with self.persistent_fields():
            self._executor = ThreadPoolExecutor(max_workers=self._MAX_WORKERS)
            self._async_variable_version_by_name: Dict[str, int] = Counter()
            self._future_by_name_and_timestamp: Dict[Tuple[str, int], Future] = {}
            self._waiters_by_future_id: Dict[int, Set[Future]] = defaultdict(set)
            self._exec_counter_by_future_id: Dict[int, int] = {}
            self._version_lock = threading.Lock()
            self._future_unwrapper = FutureUnwrapper(
                self._async_variable_version_by_name, self._future_by_name_and_timestamp
            )
        setattr(builtins, _UNWRAP_FUTURE_EXTRA_BUILTIN, self._unwrap_future)
        setattr(builtins, _FUT_TAB_EXTRA_BUILTIN, self._future_by_name_and_timestamp)

    def __del__(self):
        self._executor.shutdown()

    def _unwrap_future(self, fut):
        if isinstance(fut, Future):
            return fut.result()
        else:
            return fut

    @pyc.load_name(reentrant=True)
    def handle_load_name(self, ret, node, frame, *_, **__):
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
        async_var = stmt.targets[0].id
        unwrap_futures_expr, deps = self._future_unwrapper(node, async_var)
        unwrap_futures_code = compile(
            unwrap_futures_expr, "<pyccolo_file>", mode="eval"
        )
        with self._version_lock:
            self._async_variable_version_by_name[async_var] += 1
        current_version = self._async_variable_version_by_name[async_var]
        fut_cv = threading.Condition()
        fut = None

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
            retval = eval(unwrap_futures_code, frame.f_globals, frame.f_locals)
            # next, garbage collect the previous value
            self._waiters_by_future_id.pop(id(old_fut), None)
            self._exec_counter_by_future_id.pop(id(old_fut), None)
            self._future_by_name_and_timestamp.pop(
                (async_var, current_version - 1), None
            )
            try:
                flow = nbs()
            except:  # noqa: E722
                flow = None
            with self._version_lock:
                if self._async_variable_version_by_name[async_var] == current_version:
                    # by using 'is_outer_stmt', we can be sure
                    # that setting the global is the right thing
                    frame.f_globals[async_var] = retval
                if flow is not None:
                    with fut_cv:
                        while fut is None:
                            fut_cv.wait()
                    aliases = list(flow.aliases.get(id(fut), []))
                    for alias in aliases:
                        alias.update_obj_ref(retval)
            return retval

        ipy = get_ipython()
        current_cell = None if ipy is None else ipy.execution_count
        del ipy

        with fut_cv:
            fut = self._executor.submit(work)
            fut_cv.notify()
        self._future_by_name_and_timestamp[async_var, current_version] = fut
        if current_cell is not None:
            self._exec_counter_by_future_id[id(fut)] = current_cell
        for dep in deps:
            self._waiters_by_future_id[id(dep)].add(fut)
        return lambda: fut

    @pyc.before_stmt(when=lambda node: isinstance(node, ast.AugAssign))
    def handle_augassign(self, _ret, node, frame, *_, **__):
        async_var = node.target.id
        with self._version_lock:
            version = self._async_variable_version_by_name.get(async_var, None)
            if version is None:
                return
            else:
                fut = self._future_by_name_and_timestamp[async_var, version]
        try:
            frame.f_globals[async_var] = self._unwrap_future(fut)
        except:  # noqa: E722
            pass
