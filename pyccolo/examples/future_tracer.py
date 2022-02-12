# -*- coding: utf-8 -*-
import ast
import asyncio
import builtins
import copy
import logging
import threading
from collections import Counter
from concurrent.futures import Future
from typing import Any, Dict, List, Optional, Tuple

import pyccolo as pyc
import pyccolo.fast as fast


logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def _run_forever(loop: asyncio.BaseEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


_UNWRAP_FUTURE_EXTRA_BUILTIN = "_Xix_PYCCOLO_UNWRAP_FUTURE"
_FUT_TAB_EXTRA_BUILTIN = "_Xix_PYCCOLO_FUTURE_TABLE"


class FutureUnwrapper(ast.NodeTransformer):
    def __init__(
        self,
        async_vars: Dict[str, int],
    ) -> None:
        self._async_vars = async_vars

    def visit_Name(self, node: ast.Name):
        current_version = self._async_vars.get(node.id, None)
        if current_version is None:
            return node
        else:
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


async def _fake_coro():
    return None


class FutureTracer(pyc.BaseTracer):
    _loop = asyncio.new_event_loop()
    _thread: List[Optional[threading.Thread]] = [None]
    _async_variable_version_by_name: Dict[str, int] = Counter()
    _future_by_name_and_timestamp: Dict[Tuple[str, int], Future] = {}
    _value_by_future_id: Dict[int, Any] = {}
    _empty = object()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        setattr(builtins, _UNWRAP_FUTURE_EXTRA_BUILTIN, self._unwrap_future)
        setattr(builtins, _FUT_TAB_EXTRA_BUILTIN, self._future_by_name_and_timestamp)
        self._future_unwrapper = FutureUnwrapper(self._async_variable_version_by_name)
        if self._thread[0] is None:
            self._thread[0] = threading.Thread(target=_run_forever, args=(self._loop,))
            self._thread[0].start()
        self._current_future = asyncio.run_coroutine_threadsafe(
            _fake_coro(), loop=self._loop
        )

    @classmethod
    def clear_instance(cls):
        # TODO: replace all created futures with their materialized
        #  values in the stack frames that reference them
        if cls._loop.is_running():
            cls._loop.call_soon_threadsafe(cls._loop.stop)
            cls._thread[0].join()
            cls._thread[0] = None
        super().clear_instance()

    def _unwrap_future(self, fut):
        if isinstance(fut, Future):
            maybe_val = self._value_by_future_id.get(id(fut), self._empty)
            if maybe_val is not self._empty:
                return self._value_by_future_id[id(fut)]
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
        unwrap_futures_expr = ast.Expression(
            self._future_unwrapper.visit(copy.deepcopy(node))
        )
        unwrap_futures_code = compile(unwrap_futures_expr, "<file>", mode="eval")
        async_var = stmt.targets[0].id
        self._async_variable_version_by_name[async_var] += 1
        current_version = self._async_variable_version_by_name[async_var]

        async def async_wrapper():
            retval = eval(unwrap_futures_code, frame.f_globals, frame.f_locals)
            self._value_by_future_id[id(fut)] = retval
            old_fut = self._future_by_name_and_timestamp.pop(
                (async_var, current_version - 1), None
            )
            if old_fut is not None:
                self._value_by_future_id.pop(id(old_fut), None)
            return retval

        fut = asyncio.run_coroutine_threadsafe(async_wrapper(), loop=self._loop)
        self._current_future = fut
        self._future_by_name_and_timestamp[async_var, current_version] = fut
        return lambda: fut
