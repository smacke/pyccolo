# -*- coding: utf-8 -*-
import ast
import builtins
import functools
import inspect
import logging
import os
import sys
import textwrap
import types
from collections import defaultdict
from contextlib import contextmanager, suppress
from types import CodeType, FrameType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ContextManager,
    DefaultDict,
    Dict,
    FrozenSet,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from traitlets.config.configurable import SingletonConfigurable
from traitlets.traitlets import MetaHasTraits

from pyccolo.ast_bookkeeping import AstBookkeeper
from pyccolo.ast_rewriter import AstRewriter
from pyccolo.emit_event import (
    _TRACER_STACK,
    SANDBOX_FNAME,
    SANDBOX_FNAME_PREFIX,
    SkipAll,
    _emit_event,
    _file_passes_filter_for_event,
    _file_passes_filter_impl,
    _should_instrument_file,
    _should_instrument_file_impl,
)
from pyccolo.extra_builtins import (
    EMIT_EVENT,
    EXEC_SAVED_THUNK,
    FUNCTION_TRACING_ENABLED,
    PYCCOLO_BUILTIN_PREFIX,
    TRACE_LAMBDA,
    TRACING_ENABLED,
    make_guard_name,
)
from pyccolo.handler import HandlerSpec
from pyccolo.import_hooks import patch_meta_path_non_context
from pyccolo.predicate import Predicate
from pyccolo.syntax_augmentation import AugmentationSpec, make_syntax_augmenter
from pyccolo.trace_events import AST_TO_EVENT_MAPPING, SYS_TRACE_EVENTS, TraceEvent
from pyccolo.trace_stack import TraceStack
from pyccolo.utils import clear_keys

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


sys_gettrace = sys.gettrace
sys_settrace = sys.settrace
internal_directories = (
    os.path.dirname(os.path.dirname((lambda: 0).__code__.co_filename)),
)
Null = object()
Pass = object()
Skip = object()


PYCCOLO_DEV_MODE_ENV_VAR = "PYCCOLO_DEV_MODE"


def register_tracer_state_machine(tracer_cls: "Type[BaseTracer]") -> None:
    tracer_cls.EVENT_HANDLERS_BY_CLASS[tracer_cls] = defaultdict(
        list, tracer_cls.EVENT_HANDLERS_PENDING_REGISTRATION
    )
    tracer_cls.EVENT_HANDLERS_PENDING_REGISTRATION.clear()
    tracer_cls._MANAGER_CLASS_REGISTERED = True


class MetaTracerStateMachine(MetaHasTraits):
    def __new__(mcs, name, bases, *args, **kwargs):
        if name not in ("_InternalBaseTracer", "BaseTracer"):
            bases += (SingletonConfigurable,)
        return MetaHasTraits.__new__(mcs, name, bases, *args, **kwargs)

    def __init__(cls, *args, **kwargs):
        # we could also use __init_subclass__, but since we need a metaclass
        # anyway for other stuff, may as well keep all metaclass-y things consolidated
        super().__init__(*args, **kwargs)
        register_tracer_state_machine(cls)
        cls.defined_file = sys._getframe().f_back.f_code.co_filename

    def __call__(cls, *args, **kwargs):
        obj = MetaHasTraits.__call__(cls, *args, **kwargs)
        obj._post_init_hook_end()
        return obj

    def enable(cls, **kwargs) -> None:
        cls.instance().enable_tracing(**kwargs)  # type: ignore

    def disable(cls) -> None:
        cls.instance().disable_tracing()  # type: ignore

    def __enter__(cls) -> ContextManager:
        return cls.instance().__enter__()  # type: ignore

    def __exit__(cls, exc_type, exc_val, exc_tb):
        return cls.instance().__exit__(exc_type, exc_val, exc_tb)  # type: ignore


if TYPE_CHECKING:
    _InternalBaseTracerSuper = SingletonConfigurable
else:
    _InternalBaseTracerSuper = object


class _InternalBaseTracer(_InternalBaseTracerSuper, metaclass=MetaTracerStateMachine):
    instrument_all_files = False
    allow_reentrant_events = False
    multiple_threads_allowed = False
    requires_ast_bookkeeping = True
    should_patch_meta_path = True
    global_guards_enabled = True
    bytecode_caching_allowed = True

    ast_rewriter_cls = AstRewriter
    defined_file = ""
    sandbox_fname_counter = 0

    _MANAGER_CLASS_REGISTERED = False
    EVENT_HANDLERS_PENDING_REGISTRATION: DefaultDict[TraceEvent, List[HandlerSpec]] = (
        defaultdict(list)
    )
    EVENT_HANDLERS_BY_CLASS: Dict[
        "Type[BaseTracer]",
        DefaultDict[TraceEvent, List[HandlerSpec]],
    ] = {}

    EVENT_LOGGER = logging.getLogger("events")
    EVENT_LOGGER.setLevel(logging.WARNING)

    guards: Set[str] = set()
    local_guards_by_module_id: Dict[int, Set[str]] = defaultdict(set)
    handler_spec_by_id: Dict[int, HandlerSpec] = {}

    # shared ast bookkeeping fields
    ast_bookkeeper_by_fname: Dict[str, AstBookkeeper] = {}
    ast_node_by_id: Dict[int, ast.AST] = {}
    node_id_remapping_by_fname: Dict[str, Dict[int, int]] = {}
    containing_ast_by_id: Dict[int, ast.AST] = {}
    containing_stmt_by_id: Dict[int, ast.stmt] = {}
    parent_stmt_by_id: Dict[int, ast.stmt] = {}
    stmt_by_lineno_by_module_id: Dict[int, Dict[int, ast.stmt]] = defaultdict(dict)
    augmented_node_ids_by_spec: Dict[AugmentationSpec, Set[int]] = defaultdict(set)

    current_module: List[Optional[ast.Module]] = [None]

    def __init__(self, is_reset: bool = False):
        self._is_dev_mode = os.getenv(PYCCOLO_DEV_MODE_ENV_VAR) == "1"
        if is_reset:
            return
        if not self._MANAGER_CLASS_REGISTERED:
            raise ValueError(
                f"class not registered; use the `{register_tracer_state_machine.__name__}` "
                + "decorator on the subclass"
            )
        super().__init__()
        self._has_fancy_sys_tracing = sys.version_info >= (3, 7)
        self._event_handlers: DefaultDict[TraceEvent, List[HandlerSpec]] = defaultdict(
            list
        )
        self._handler_names: Set[str] = set()
        events_with_registered_handlers = set()
        for clazz in reversed(self.__class__.mro()):
            for evt, handlers in self.EVENT_HANDLERS_BY_CLASS.get(clazz, {}).items():
                for handler_spec in handlers:
                    predicate = handler_spec.predicate
                    if hasattr(predicate, "condition"):
                        condition: Optional[Callable[..., bool]] = getattr(
                            self,
                            getattr(predicate.condition, "__name__", "<empty>"),
                            None,
                        )
                        if condition is not None:
                            predicate.condition = condition
                self._event_handlers[evt].extend(handlers)
                self._handler_names |= {handler[0].__name__ for handler in handlers}
                if not issubclass(BaseTracer, clazz) and len(handlers) > 0:
                    events_with_registered_handlers.add(evt)
        self.events_with_registered_handlers: FrozenSet[TraceEvent] = frozenset(
            events_with_registered_handlers
        )
        self._ctx: Optional[ContextManager] = None
        self._tracing_enabled_files: Set[str] = {self.defined_file}
        self._current_sandbox_fname: str = SANDBOX_FNAME
        self._saved_thunk: Optional[Union[str, ast.AST]] = None
        self._is_tracing_enabled = False
        self._is_tracing_hard_disabled = False
        self.existing_tracer = sys_gettrace()
        self.sys_tracer = self._make_composed_tracer(self.existing_tracer)
        self._num_sandbox_calls_seen: int = 0

        self._transient_fields: Set[str] = set()
        self._persistent_fields: Set[str] = set()
        self._manual_persistent_fields: Set[str] = set()
        self._post_init_hook_start()

    @classmethod
    def remove_bookkeeping(
        cls, bookkeeper: AstBookkeeper, module_id: Optional[int]
    ) -> None:
        clear_keys(cls.ast_node_by_id, bookkeeper.ast_node_by_id)
        clear_keys(cls.containing_ast_by_id, bookkeeper.containing_ast_by_id)
        clear_keys(cls.containing_stmt_by_id, bookkeeper.containing_stmt_by_id)
        clear_keys(cls.parent_stmt_by_id, bookkeeper.parent_stmt_by_id)
        if module_id is not None:
            clear_keys(
                cls.stmt_by_lineno_by_module_id[module_id], bookkeeper.stmt_by_lineno
            )
        for spec, node_ids in cls.augmented_node_ids_by_spec.items():
            clear_keys(node_ids, bookkeeper.ast_node_by_id)

    @classmethod
    def add_bookkeeping(cls, bookkeeper: AstBookkeeper, module_id: int) -> None:
        cls.ast_node_by_id.update(bookkeeper.ast_node_by_id)
        cls.containing_ast_by_id.update(bookkeeper.containing_ast_by_id)
        cls.containing_stmt_by_id.update(bookkeeper.containing_stmt_by_id)
        cls.parent_stmt_by_id.update(bookkeeper.parent_stmt_by_id)
        cls.stmt_by_lineno_by_module_id[module_id].update(bookkeeper.stmt_by_lineno)

    @classmethod
    def reset_bookkeeping(cls) -> None:
        cls.ast_node_by_id.clear()
        cls.containing_ast_by_id.clear()
        cls.containing_stmt_by_id.clear()
        cls.parent_stmt_by_id.clear()
        cls.stmt_by_lineno_by_module_id.clear()
        for node_ids in cls.augmented_node_ids_by_spec.values():
            node_ids.clear()

    @property
    def has_sys_trace_events(self):
        return any(
            evt in self.events_with_registered_handlers for evt in SYS_TRACE_EVENTS
        )

    @property
    def syntax_augmentation_specs(self) -> List[AugmentationSpec]:
        return []

    def get_augmentations(
        self, node_id: Union[ast.AST, int]
    ) -> FrozenSet[AugmentationSpec]:
        if isinstance(node_id, ast.AST):
            node_id = id(node_id)
        augs = []
        for aug, node_ids in self.augmented_node_ids_by_spec.items():
            if node_id in node_ids:
                augs.append(aug)
        return frozenset(augs)

    @classmethod
    def make_sandbox_fname(cls) -> str:
        cls.sandbox_fname_counter += 1
        return f"{SANDBOX_FNAME_PREFIX}-{cls.sandbox_fname_counter}>"

    @property
    def is_tracing_enabled(self) -> bool:
        return self._is_tracing_enabled

    def _post_init_hook_start(self):
        self._persistent_fields = set(self.__dict__.keys())

    def _post_init_hook_end(self):
        self._transient_fields = (
            set(self.__dict__.keys())
            - self._persistent_fields
            - self._manual_persistent_fields
        )

    @contextmanager
    def persistent_fields(self) -> Generator[None, None, None]:
        current_fields = set(self.__dict__.keys())
        saved_fields = {}
        for field in self._manual_persistent_fields:
            if field in current_fields:
                saved_fields[field] = self.__dict__[field]
        yield
        self._manual_persistent_fields = (
            self.__dict__.keys() - current_fields
        ) | saved_fields.keys()
        for field, val in saved_fields.items():
            self.__dict__[field] = val

    def reset(self):
        for field in self._transient_fields:
            del self.__dict__[field]
        self.__init__(is_reset=True)

    @classmethod
    def _make_guard_name(cls, guard: Union[str, int, ast.AST]) -> str:
        if isinstance(guard, str):
            guard_name = guard
        else:
            guard_name = make_guard_name(guard)
        assert guard_name in cls.guards
        return guard_name

    @classmethod
    def activate_guard(cls, guard: Union[str, int, ast.AST]) -> None:
        setattr(builtins, cls._make_guard_name(guard), False)

    @classmethod
    def deactivate_guard(cls, guard: Union[str, int, ast.AST]) -> None:
        setattr(builtins, cls._make_guard_name(guard), True)

    @classmethod
    def register_local_guard(cls, guard: str) -> None:
        assert cls.current_module[0] is not None
        cls.local_guards_by_module_id[id(cls.current_module[0])].add(guard)

    def should_propagate_handler_exception(
        self, evt: TraceEvent, exc: Exception
    ) -> bool:
        return False

    def _handle_skipall_emit_return(self, event, old_ret):
        if event in (TraceEvent.call, TraceEvent.exception):
            return (SkipAll, self.sys_tracer)
        else:
            return (SkipAll, old_ret)

    def _handle_normal_emit_return(self, event, old_ret, new_ret):
        should_break = new_ret is Skip
        if new_ret is None or new_ret is Skip:
            if event in (TraceEvent.call, TraceEvent.exception):
                new_ret = self.sys_tracer
            else:
                new_ret = old_ret
        elif new_ret is Null:
            new_ret = None
        return new_ret, should_break

    def _emit_event(
        self,
        evt: Union[str, TraceEvent],
        node_id: Optional[int],
        frame: FrameType,
        reentrant_handlers_only: bool = False,
        **kwargs: Any,
    ):
        try:
            if self._is_tracing_hard_disabled:
                return kwargs.get("ret")
            event = evt if isinstance(evt, TraceEvent) else TraceEvent(evt)
            guards_by_spec_id = kwargs.get("guards_by_handler_spec_id")
            for spec in self._event_handlers.get(event, []):
                if reentrant_handlers_only and not spec.reentrant:
                    continue
                guard_for_spec = (
                    None
                    if guards_by_spec_id is None
                    else guards_by_spec_id.get(id(spec))
                )
                if guard_for_spec is not None and frame.f_globals.get(
                    guard_for_spec, False
                ):
                    continue
                old_ret = kwargs.pop("ret", None)
                try:
                    node_id_or_node = (
                        node_id
                        if spec.use_raw_node_id
                        else self.ast_node_by_id.get(node_id or -1)
                    )
                    if (
                        spec.predicate is Predicate.TRUE
                        or spec.predicate.static
                        or spec.predicate.dynamic_call(node_id_or_node or -1)
                    ):
                        new_ret = spec.handler(
                            self,
                            old_ret,
                            node_id_or_node,
                            frame,
                            event,
                            guard_for_spec,
                            **kwargs,
                        )
                    else:
                        new_ret = None
                except Exception as exc:
                    if self.should_propagate_handler_exception(event, exc):
                        raise exc
                    elif self._is_dev_mode:
                        logger.exception("An exception while handling evt %s", event)
                    new_ret = None
                if new_ret is SkipAll:
                    return self._handle_skipall_emit_return(event, old_ret)
                else:
                    new_ret, should_break = self._handle_normal_emit_return(
                        event, old_ret, new_ret
                    )
                kwargs["ret"] = new_ret
                if event == TraceEvent.before_stmt:
                    self._saved_thunk = new_ret
                if should_break:
                    break
            return kwargs.get("ret")
        except KeyboardInterrupt as ki:
            self._disable_tracing(check_enabled=False)
            raise ki.with_traceback(None)

    def make_stack(self):
        return TraceStack(self)

    def _call_existing_tracer(
        self, existing_tracer, frame: FrameType, evt: str, arg: Any, **kwargs
    ):  # pragma: no cover
        if existing_tracer is None:
            return None
        orig_sys_tracer = sys_gettrace()
        existing_ret = existing_tracer(frame, evt, arg, **kwargs)
        if sys_gettrace() is not orig_sys_tracer:
            # to deal with the existing tracer messing with things
            sys_settrace(orig_sys_tracer)
        return existing_ret

    def _make_composed_tracer(self, existing_tracer):  # pragma: no cover
        @functools.wraps(self._sys_tracer)
        def _composed_tracer(frame: FrameType, evt: str, arg: Any, **kwargs):
            __debuggerskip__ = True  # noqa: F841
            if self._is_tracing_enabled:
                my_ret = self._sys_tracer(frame, evt, arg, **kwargs)
            else:
                my_ret = None
            if isinstance(my_ret, tuple) and len(my_ret) > 1 and my_ret[0] is SkipAll:
                return my_ret[1]
            existing_ret = self._call_existing_tracer(
                existing_tracer, frame, evt, arg, **kwargs
            )
            if evt == "call":
                if my_ret is not None and existing_ret is not None:
                    return self._make_composed_tracer(existing_ret)
                elif my_ret is None:
                    return existing_ret
            return my_ret

        return _composed_tracer

    def _enable_tracing(self, check_disabled=True, existing_tracer=None):
        if check_disabled:
            assert not self._is_tracing_enabled
        self._is_tracing_enabled = True
        if self.has_sys_trace_events:
            self.existing_tracer = existing_tracer or sys_gettrace()
            self.sys_tracer = self._make_composed_tracer(self.existing_tracer)
            sys_settrace(self.sys_tracer)
        setattr(builtins, FUNCTION_TRACING_ENABLED, True)
        setattr(builtins, TRACING_ENABLED, True)

    def _disable_tracing(self, check_enabled=True):
        has_sys_trace_events = self.has_sys_trace_events
        if check_enabled:
            assert self._is_tracing_enabled
            assert not has_sys_trace_events or sys_gettrace() is self.sys_tracer
        self._is_tracing_enabled = False
        if has_sys_trace_events and sys_gettrace() is not None:
            sys_settrace(self.existing_tracer)
        setattr(builtins, FUNCTION_TRACING_ENABLED, False)
        if len(_TRACER_STACK) == 0:
            setattr(builtins, TRACING_ENABLED, False)

    def _patch_sys_settrace_non_context(self) -> Callable:
        import threading

        original_sys_gettrace = sys.gettrace
        original_sys_settrace = sys.settrace
        orig_thread = threading.current_thread()
        existing_tracer = self.existing_tracer

        def cleanup_callback():
            sys.gettrace = original_sys_gettrace
            sys.settrace = original_sys_settrace

        def patched_sys_settrace(trace_func):  # pragma: no cover
            if threading.current_thread() is not orig_thread:
                if trace_func is None:
                    return sys_settrace(None)
                else:
                    return sys_settrace(trace_func)
            # called by third-party tracers
            self.existing_tracer = trace_func
            if self._is_tracing_enabled:
                if trace_func is None:
                    self._disable_tracing()
                self._enable_tracing(check_disabled=False, existing_tracer=trace_func)
            else:
                original_sys_settrace(trace_func)

        def patched_sys_gettrace():
            return existing_tracer

        sys.gettrace = patched_sys_gettrace
        sys.settrace = patched_sys_settrace
        return cleanup_callback

    @contextmanager
    def _patch_sys_settrace(self) -> Generator[None, None, None]:
        cleanup_callback = None
        try:
            cleanup_callback = self._patch_sys_settrace_non_context()
            yield
        finally:
            if cleanup_callback is not None:
                cleanup_callback()

    def file_passes_filter_for_event(self, evt: str, filename: str) -> bool:
        return _file_passes_filter_for_event(self, evt, filename)

    def should_instrument_file(self, filename: str) -> bool:
        return _should_instrument_file(self, filename)

    def _should_instrument_file_impl(self, filename: str) -> bool:
        return _should_instrument_file_impl(self, filename)

    def _file_passes_filter_impl(
        self, evt: str, filename: str, is_reentrant: bool = False
    ) -> bool:
        return _file_passes_filter_impl(self, evt, filename, is_reentrant=is_reentrant)

    def make_ast_rewriter(
        self, path: str, module_id: Optional[int] = None
    ) -> AstRewriter:
        return self.ast_rewriter_cls(_TRACER_STACK, path, module_id=module_id)

    def make_syntax_augmenters(self, ast_rewriter: AstRewriter) -> List[Callable]:
        return [
            make_syntax_augmenter(ast_rewriter, spec)
            for spec in self.syntax_augmentation_specs
        ]

    @contextmanager
    def tracing_enabled(self, **kwargs) -> Generator[None, None, None]:
        with self.tracing_context(disabled=False, **kwargs):
            yield

    def enable_tracing(self, **kwargs) -> None:
        self.__enter__(**kwargs)

    def disable_tracing(self) -> None:
        self.__exit__(None, None, None)

    def __enter__(self, **kwargs) -> ContextManager:
        assert self._ctx is None
        self._ctx = self.tracing_enabled(**kwargs)
        return self._ctx.__enter__()  # type: ignore

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._ctx is not None
        ctx = self._ctx
        self._ctx = None
        return ctx.__exit__(exc_type, exc_val, exc_tb)

    @contextmanager
    def tracing_disabled(self, **kwargs) -> Generator[None, None, None]:
        orig_num_sandbox_calls_seen = self._num_sandbox_calls_seen
        self._num_sandbox_calls_seen = 2
        try:
            with self.tracing_context(disabled=True, **kwargs):
                yield
        finally:
            self._num_sandbox_calls_seen = orig_num_sandbox_calls_seen

    def instrumented(self, f: Callable) -> Callable:
        f_defined_file = f.__code__.co_filename
        with self.tracing_disabled():
            code = ast.parse(textwrap.dedent(inspect.getsource(f)))
            code.body[0] = self.make_ast_rewriter(f.__code__.co_filename).visit(
                code.body[0]
            )
            compiled: types.CodeType = compile(code, f.__code__.co_filename, "exec")
            for const in compiled.co_consts:
                if (
                    isinstance(const, types.CodeType)
                    and const.co_name == f.__code__.co_name
                ):
                    f.__code__ = const
                    break

        @functools.wraps(f)
        def instrumented_f(*args, **kwargs):
            with self.tracing_enabled(tracing_enabled_file=f_defined_file):
                return f(*args, **kwargs)

        return instrumented_f

    def __call__(self, code: Union[str, ast.Module, ast.stmt, Callable]):
        if isinstance(code, (str, ast.AST)):
            return self.exec(code, num_extra_lookback_frames=1)
        else:
            return self.instrumented(code)

    def __getitem__(self, code: Union[str, ast.Module, ast.stmt, Callable]):
        return self(code)

    def enter_tracing_hook(self) -> None:
        pass

    def exit_tracing_hook(self) -> None:
        pass

    def _static_init_module_impl(self, node: ast.Module) -> None:
        self.current_module[0] = node
        self.static_init_module(node)

    def static_init_module(self, node: ast.Module) -> None:
        pass

    def _make_tracing_context_cleanup_callback(self):
        orig_num_sandbox_calls_seen = self._num_sandbox_calls_seen
        orig_hard_disabled = self._is_tracing_hard_disabled
        orig_exec_saved_thunk = getattr(builtins, EXEC_SAVED_THUNK, None)
        orig_sandbox_fname = self._current_sandbox_fname
        orig_tracing_enabled_files = self._tracing_enabled_files

        def cleanup(should_push: bool, will_enable_tracing: bool) -> None:
            self._tracing_enabled_files = orig_tracing_enabled_files
            self._current_sandbox_fname = orig_sandbox_fname
            self._is_tracing_hard_disabled = orig_hard_disabled
            self._num_sandbox_calls_seen = orig_num_sandbox_calls_seen

            if should_push:
                del _TRACER_STACK[-1]
            if will_enable_tracing:
                self._disable_tracing(check_enabled=False)
            if should_push:
                self.exit_tracing_hook()

            if len(_TRACER_STACK) == 0:
                for extra_builtin in {
                    EMIT_EVENT,
                    EXEC_SAVED_THUNK,
                    TRACE_LAMBDA,
                    TRACING_ENABLED,
                } | self.guards:
                    if hasattr(builtins, extra_builtin):
                        delattr(builtins, extra_builtin)
            elif orig_exec_saved_thunk is not None:
                setattr(builtins, EXEC_SAVED_THUNK, orig_exec_saved_thunk)

        return cleanup

    @contextmanager
    def tracing_context(
        self, disabled: bool = False, tracing_enabled_file: Optional[str] = None
    ) -> Generator[None, None, None]:
        cleanup_callback = None
        try:
            cleanup_callback = self.tracing_non_context(
                disabled=disabled, tracing_enabled_file=tracing_enabled_file
            )
            yield
        finally:
            if cleanup_callback is not None:
                cleanup_callback()

    def tracing_non_context(
        self,
        disabled: bool = False,
        tracing_enabled_file: Optional[str] = None,
        do_patch_meta_path: Optional[bool] = None,
    ) -> Callable:
        cleanup_callback_impl = self._make_tracing_context_cleanup_callback()
        should_push = self not in _TRACER_STACK
        self._is_tracing_hard_disabled = disabled
        will_enable_tracing = (
            not self._is_tracing_hard_disabled and not self._is_tracing_enabled
        )

        def first_cleanup_callback():
            return cleanup_callback_impl(should_push, will_enable_tracing)

        all_cleanup_callbacks = [first_cleanup_callback]

        def cleanup_callback():
            for cleanup in reversed(all_cleanup_callbacks):
                cleanup()

        if tracing_enabled_file is not None:
            self._current_sandbox_fname = tracing_enabled_file
            self._tracing_enabled_files = self._tracing_enabled_files | {
                tracing_enabled_file
            }
        if getattr(builtins, EMIT_EVENT, None) is not _emit_event:
            setattr(builtins, EMIT_EVENT, _emit_event)
            for guard in self.guards:
                self.deactivate_guard(guard)
        if not hasattr(builtins, TRACING_ENABLED):
            setattr(builtins, TRACING_ENABLED, False)
        if not hasattr(builtins, FUNCTION_TRACING_ENABLED):
            setattr(builtins, FUNCTION_TRACING_ENABLED, False)
        setattr(builtins, EXEC_SAVED_THUNK, self.exec_saved_thunk)
        setattr(builtins, TRACE_LAMBDA, self.trace_lambda)
        if do_patch_meta_path is None:
            do_patch_meta_path = self.should_patch_meta_path and len(_TRACER_STACK) == 0
        if should_push:
            _TRACER_STACK.append(self)  # type: ignore
        do_patch_sys_settrace = self.has_sys_trace_events and will_enable_tracing
        if do_patch_meta_path:
            all_cleanup_callbacks.append(patch_meta_path_non_context(_TRACER_STACK))
        if do_patch_sys_settrace:
            all_cleanup_callbacks.append(self._patch_sys_settrace_non_context())
        if will_enable_tracing:
            self._enable_tracing()
        if should_push:
            self.enter_tracing_hook()
        return cleanup_callback

    def preprocess(self, code: str, rewriter: AstRewriter) -> str:
        for augmenter in self.make_syntax_augmenters(rewriter):
            code = augmenter(code)
        return code

    def parse(self, code: str, mode="exec") -> Union[ast.Module, ast.Expression]:
        rewriter = self.make_ast_rewriter(self.make_sandbox_fname())
        for tracer in _TRACER_STACK:
            code = tracer.preprocess(code, rewriter)
        return rewriter.visit(ast.parse(code, mode=mode))

    def exec_raw(
        self,
        code: Union[ast.Module, ast.Expression, str],
        global_env: dict,
        local_env: dict,
        filename: str,
        instrument: bool = True,
        do_eval: bool = False,
    ) -> Any:
        if filename is None:
            filename = self.make_sandbox_fname()
        with (
            self.tracing_context(
                disabled=self._is_tracing_hard_disabled,
                tracing_enabled_file=filename,
            )
            if instrument
            else suppress()
        ):
            if isinstance(code, str):
                code = textwrap.dedent(code).strip()
                code = self.parse(code)
            if instrument:
                code = self.make_ast_rewriter(path=filename).visit(code)
            code_obj = compile(code, filename, "eval" if do_eval else "exec")
            if do_eval:
                self._num_sandbox_calls_seen = 2
                return eval(code_obj, global_env, local_env)
            else:
                return exec(code_obj, global_env, local_env)

    @staticmethod
    def _get_environments(
        global_env: Optional[dict],
        local_env: Optional[dict],
        num_extra_lookback_frames: int,
    ) -> Tuple[dict, dict]:
        if global_env is None or local_env is None:
            frame = sys._getframe().f_back
            assert frame is not None
            for _ in range(num_extra_lookback_frames):
                frame = frame.f_back
                assert frame is not None
            if global_env is None:
                global_env = frame.f_globals
            if local_env is None:
                local_env = frame.f_locals
        return global_env, local_env

    def eval(
        self,
        code: Union[str, ast.expr, ast.Expression],
        global_env: Optional[dict] = None,
        local_env: Optional[dict] = None,
        *,
        instrument: bool = True,
        filename: Optional[str] = None,
        num_extra_lookback_frames: int = 0,
    ) -> Any:
        if filename is None:
            filename = self.make_sandbox_fname()
        global_env, local_env = self._get_environments(
            global_env, local_env, num_extra_lookback_frames + 1
        )
        with (
            self.tracing_context(
                disabled=self._is_tracing_hard_disabled,
                tracing_enabled_file=filename,
            )
            if instrument
            else suppress()
        ):
            visited = False
            if isinstance(code, str):
                if instrument:
                    visited = True
                    code = cast(ast.Expression, self.parse(code, mode="eval"))
                else:
                    code = cast(ast.Expression, ast.parse(code, mode="eval"))
            if not isinstance(code, ast.Expression):
                code = ast.Expression(code)
            if instrument and not visited:
                code = self.make_ast_rewriter(path=filename).visit(code)
            return self.exec_raw(
                code,  # type: ignore
                global_env=global_env,
                local_env=local_env,
                filename=filename,
                instrument=False,
                do_eval=True,
            )

    def exec(
        self,
        code: Union[str, ast.Module, ast.stmt],
        global_env: Optional[dict] = None,
        local_env: Optional[dict] = None,
        *,
        instrument: bool = True,
        filename: Optional[str] = None,
        num_extra_lookback_frames: int = 0,
    ) -> Dict[str, Any]:
        if filename is None:
            filename = self.make_sandbox_fname()
        global_env, local_env = self._get_environments(
            global_env, local_env, num_extra_lookback_frames + 1
        )
        # pytest inserts variables prepended with "@"; we don't want these
        args_to_use = [
            k for k in local_env.keys() if not k.startswith("@") and k != "__"
        ]
        if len(args_to_use) > 0:
            sandbox_args = ", ".join(["*"] + args_to_use + ["**__"])
        else:
            sandbox_args = "**__"
        env_name = f"{PYCCOLO_BUILTIN_PREFIX}_pyccolo_local_env"
        fun_name = f"{PYCCOLO_BUILTIN_PREFIX}_pyccolo_sandbox"
        sandboxed_code: Union[ast.Module, str] = textwrap.dedent(
            f"""
            {env_name} = dict(locals())
            def {fun_name}({sandbox_args}):
                return locals()
            {env_name} = {fun_name}(**{env_name})
            {env_name}.pop("__", None)
            {env_name}.pop("builtins", None)
            """
        ).strip()
        sandboxed_code = ast.parse(cast(str, sandboxed_code), filename, "exec")
        with (
            self.tracing_context(
                disabled=self._is_tracing_hard_disabled,
                tracing_enabled_file=filename,
            )
            if instrument
            else suppress()
        ):
            visited = False
            if isinstance(code, str):
                code = textwrap.dedent(code).strip()
                if instrument:
                    visited = True
                    code = cast(ast.Module, self.parse(code))
                else:
                    code = cast(ast.Module, ast.parse(code))
            if not isinstance(code, ast.Module):
                assert isinstance(code, ast.stmt)
                if sys.version_info < (3, 8):
                    code = ast.Module([code])
                else:
                    code = ast.Module([code], [])
            if instrument and not visited:
                code = self.make_ast_rewriter(path=filename).visit(code)
            # prepend the stuff before "return locals()"
            fundef: ast.FunctionDef = cast(ast.FunctionDef, sandboxed_code.body[1])
            if isinstance(code, ast.Module):
                code_body: List[ast.stmt] = code.body
            else:
                assert isinstance(code, ast.stmt)
                code_body = [code]
            fundef.body = code_body + fundef.body
            self.exec_raw(
                sandboxed_code,
                global_env=global_env,
                local_env=local_env,
                filename=filename,
                instrument=False,
            )
        return local_env.pop(env_name)

    def trace_lambda(self, lam: Callable[..., Any]) -> Callable[..., Any]:
        # for now, this is primarily so that we can distinguish between
        # lambdas that we generate vs that user generates
        code: CodeType = lam.__code__
        assert code.co_name == "<lambda>"
        if sys.version_info >= (3, 8):
            lam.__code__ = code.replace(co_name="<traced_lambda>")
        else:
            # replace(...) not available for older python but CodeType
            # constructor is stable at least
            lam.__code__ = CodeType(
                code.co_argcount,
                code.co_kwonlyargcount,
                code.co_nlocals,
                code.co_stacksize,
                code.co_flags,
                code.co_code,
                code.co_consts,
                code.co_names,
                code.co_varnames,
                code.co_filename,
                "<traced_lambda>",
                code.co_firstlineno,
                code.co_lnotab,
                code.co_freevars,
                code.co_cellvars,
            )
        return lam

    def exec_saved_thunk(self):
        assert self._saved_thunk is not None
        thunk, self._saved_thunk = self._saved_thunk, None
        if thunk is not Pass:
            return self.exec(thunk, instrument=False, num_extra_lookback_frames=1)

    def execute(self, *args, **kwargs):
        return self.exec(*args, **kwargs)

    def _should_attempt_to_reenable_tracing(self, frame: FrameType) -> bool:
        return NotImplemented

    def _sys_tracer(self, frame: FrameType, evt: str, arg: Any, **__):
        if not self._file_passes_filter_impl(evt, frame.f_code.co_filename):
            return None
        if evt == "call" and frame.f_code.co_filename == self.defined_file:
            func_name = frame.f_code.co_name
            if func_name in self._handler_names and not self.allow_reentrant_events:
                return None

        if self._has_fancy_sys_tracing and evt == "call":
            orig_trace_lines = frame.f_trace_lines  # type: ignore
            orig_trace_opcodes = frame.f_trace_opcodes  # type: ignore
            frame.f_trace_lines = (  # type: ignore
                TraceEvent.line not in self.events_with_registered_handlers
            )
            frame.f_trace_opcodes = (  # type: ignore
                TraceEvent.opcode in self.events_with_registered_handlers
            )
            try:
                return self._emit_event(evt, None, frame, ret=arg)
            finally:
                frame.f_trace_lines = orig_trace_lines  # type: ignore
                frame.f_trace_opcodes = orig_trace_opcodes  # type: ignore
        else:
            return self._emit_event(evt, None, frame, ret=arg)

    if TYPE_CHECKING:
        TracerT = TypeVar("TracerT", bound="_InternalBaseTracer")

        @classmethod
        def instance(cls: Type[TracerT], *args, **kwargs) -> TracerT: ...

        @classmethod
        def clear_instance(cls) -> None: ...


def register_handler(
    event: Union[
        Union[TraceEvent, Type[ast.AST]], Tuple[Union[TraceEvent, Type[ast.AST]], ...]
    ],
    when: Optional[Union[Callable[..., bool], Predicate]] = None,
    reentrant: bool = False,
    use_raw_node_id: bool = False,
    guard: Optional[Callable[[ast.AST], str]] = None,
    exempt_from_guards: bool = False,
):
    events = event if isinstance(event, tuple) else (event,)
    when = Predicate.TRUE if when is None else when
    if isinstance(when, Predicate):
        pred: Predicate = when
    else:
        pred = Predicate(when, use_raw_node_id=use_raw_node_id)  # type: ignore
    pred.use_raw_node_id = use_raw_node_id

    if TraceEvent.opcode in events and sys.version_info < (3, 7):
        raise ValueError("can't trace opcodes on Python < 3.7")

    def _inner_registrar(handler):
        for evt in events:
            handler_spec = HandlerSpec(
                handler, use_raw_node_id, reentrant, pred, guard, exempt_from_guards
            )
            _InternalBaseTracer.EVENT_HANDLERS_PENDING_REGISTRATION[
                (
                    AST_TO_EVENT_MAPPING[evt]
                    if type(evt) is type and issubclass(evt, ast.AST)
                    else evt
                )
            ].append(handler_spec)
            _InternalBaseTracer.handler_spec_by_id[id(handler_spec)] = handler_spec
        return handler

    return _inner_registrar


def __event_call__(self, handler=None, **kwargs):
    if handler is None:

        def _register_func(_handler):
            return register_handler(self, **kwargs)(_handler)

        return _register_func
    else:
        if len(kwargs) > 0:
            raise ValueError(
                "keyword arguments not supported for simple handler registration"
            )
        return register_handler(self)(handler)


TraceEvent.__call__ = __event_call__  # type: ignore


def register_raw_handler(
    event: Union[
        Union[TraceEvent, Type[ast.AST]], Tuple[Union[TraceEvent, Type[ast.AST]], ...]
    ],
    **kwargs,
):
    return register_handler(event, use_raw_node_id=True, **kwargs)


def skip_when_tracing_disabled(handler):
    @functools.wraps(handler)
    def skipping_handler(self, *args, **kwargs):
        if not self._is_tracing_enabled:
            return
        return handler(self, *args, **kwargs)

    return skipping_handler


def register_universal_handler(handler):
    return register_handler(tuple(evt for evt in TraceEvent))(handler)


class BaseTracer(_InternalBaseTracer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved_expr_stmt_ret: Optional[Any] = None
        self._saved_slice: Optional[Any] = None

    @register_raw_handler(TraceEvent.after_stmt, reentrant=True)
    def _save_expr_stmt_ret_for_later(self, ret_expr: Any, *_, **__) -> None:
        self._saved_expr_stmt_ret = ret_expr

    @register_raw_handler(TraceEvent._load_saved_expr_stmt_ret, reentrant=True)
    def _load_saved_expr_stmt_ret(self, *_, **__) -> Any:
        ret = self._saved_expr_stmt_ret
        self._saved_expr_stmt_ret = None
        return ret

    @register_raw_handler(
        (
            TraceEvent.before_subscript_load,
            TraceEvent.before_subscript_store,
            TraceEvent.before_subscript_del,
        ),
        reentrant=True,
    )
    def _save_slice_for_later(self, *_, attr_or_subscript: Any, **__):
        self._saved_slice = attr_or_subscript

    @register_raw_handler(TraceEvent._load_saved_slice, reentrant=True)
    def _load_saved_slice(self, *_, **__):
        ret = self._saved_slice
        self._saved_slice = None
        return ret

    @classmethod
    def stmt_only_has_ancestor_types(
        cls,
        node_or_id: Union[int, ast.AST],
        ancestor_types: Tuple[Type[ast.AST], ...],
    ):
        node_id = node_or_id if isinstance(node_or_id, int) else id(node_or_id)
        containing_stmt = cls.containing_stmt_by_id.get(node_id)
        parent_stmt = cls.parent_stmt_by_id.get(
            node_id if containing_stmt is None else id(containing_stmt)
        )
        while parent_stmt is not None and isinstance(parent_stmt, ancestor_types):
            parent_stmt = cls.parent_stmt_by_id.get(id(parent_stmt))
        return parent_stmt is None or isinstance(parent_stmt, ast.Module)

    @classmethod
    def is_outer_stmt(cls, node_or_id, exclude_outer_stmt_types=None):
        ancestor_types = tuple(
            {ast.If, ast.Try, ast.With, ast.AsyncWith}
            - (exclude_outer_stmt_types or set())
        )
        return cls.stmt_only_has_ancestor_types(node_or_id, ancestor_types)

    @classmethod
    def is_initial_frame_stmt(cls, node_or_id: Union[int, ast.AST]):
        return cls.stmt_only_has_ancestor_types(
            node_or_id,
            (
                ast.If,
                ast.Try,
                ast.With,
                ast.AsyncWith,
                ast.For,
                ast.AsyncFor,
                ast.While,
            ),
        )


class NoopTracer(BaseTracer):
    should_patch_meta_path = False
    global_guards_enabled = False

    def file_passes_filter_for_event(self, evt: str, filename: str) -> bool:
        return False
