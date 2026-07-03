# -*- coding: utf-8 -*-
import ast
import builtins
import contextvars
import functools
import inspect
import linecache
import logging
import os
import sys
import textwrap
import warnings
from collections import defaultdict
from contextlib import contextmanager, suppress
from types import CodeType, FrameType, FunctionType
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
    overload,
)

from traitlets.config.configurable import SingletonConfigurable
from traitlets.traitlets import MetaHasTraits

from pyccolo.ast_bookkeeping import AstBookkeeper, BookkeepingVisitor
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
    NAME_ERROR_MATCHES,
    PYCCOLO_BUILTIN_PREFIX,
    TRACE_LAMBDA,
    TRACING_ENABLED,
    make_guard_name,
    name_error_matches_prefix,
)
from pyccolo.handler import HandlerSpec
from pyccolo.import_hooks import patch_meta_path_non_context
from pyccolo.predicate import Predicate
from pyccolo.stmt_mapper import StatementMapper
from pyccolo.syntax_augmentation import (
    AugmentationSpec,
    AugmentationType,
    Edit,
    Position,
    Range,
    _line_starts,
    line_col_of,
    offset_of,
    remap_through_edits,
    replace_paired_delimiters_and_get_augmented_positions,
    replace_tokens_and_get_augmented_positions,
)
from pyccolo.trace_events import (
    AST_TO_EVENT_MAPPING,
    EVT_TO_EVENT_MAPPING,
    SYS_TRACE_EVENTS,
    TraceEvent,
)
from pyccolo.trace_stack import TraceStack
from pyccolo.utils import clear_keys, copy_function_with_code

if TYPE_CHECKING:
    from syntax_augmentation import CodeLines

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


HIDE_PYCCOLO_FRAME = "__hide_pyccolo_frame__"
PYCCOLO_DEV_MODE_ENV_VAR = "PYCCOLO_DEV_MODE"
TRACED_LAMBDA_NAME = "<traced_lambda>"


# Set while a ``pure=True`` transform/untransform runs. A ``ContextVar`` (not a
# plain global or instance attr) so concurrent transforms on different threads /
# async tasks -- e.g. a linter transforming on a background thread while the
# kernel executes on another -- don't clobber each other's flag.
_PURE_TRANSFORM: "contextvars.ContextVar[bool]" = contextvars.ContextVar(
    "pyccolo_pure_transform", default=False
)


def is_pure_transform() -> bool:
    """True while a ``pure=True`` transform/untransform is running on this
    thread / async context. Cooperating ``CustomRewrite.rewrite`` /
    paired-delimiter ``emit`` callbacks must not mutate execution-relevant state
    (e.g. process-global registries the runtime later reads) when this is set --
    the transformed code is being inspected, not executed."""
    return _PURE_TRANSFORM.get()


def _find_lambda(node: ast.AST, argcount: int) -> "Optional[ast.Lambda]":
    """Find the ``ast.Lambda`` in ``node`` with the given positional arg count
    (used to recover a lambda's body from the statement that ``getsource`` returns)."""
    found: "Optional[ast.Lambda]" = None
    for child in ast.walk(node):
        if isinstance(child, ast.Lambda):
            n_args = len(getattr(child.args, "posonlyargs", [])) + len(child.args.args)
            if n_args == argcount:
                found = child
    return found


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
    # When True, ``instrumented`` will weave a bare ``lambda`` (recovered from the
    # statement that defines it) by lifting it into a synthetic ``def``; otherwise
    # only ``def``/``async def`` targets can be (re)instrumented from source.
    instrument_lambdas = False
    # When True, ``eval`` registers the (pre-instrumentation) source of sandbox
    # code in ``linecache`` so ``inspect.getsource`` and tracebacks work for code
    # with no on-disk source -- e.g. a pipescript ``|>`` pipe lambda. Off by
    # default: it unparses + caches on every eval, which is wasteful in hot paths.
    keep_sandbox_source = False

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
    additional_ast_bookkeeping: Dict[str, Union[Dict[int, Any], Set[int]]] = {}

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
        self.last_applied_specs: List[AugmentationSpec] = []
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
        for extra_bookkeeping in cls.additional_ast_bookkeeping.values():
            clear_keys(extra_bookkeeping, bookkeeper.ast_node_by_id)

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
        for extra_bookkeeping in cls.additional_ast_bookkeeping.values():
            extra_bookkeeping.clear()

    @contextmanager
    def register_additional_ast_bookkeeping(self):
        original_state = set(self.__dict__.keys())
        yield
        for additional_item_name in set(self.__dict__.keys() - original_state):
            self.additional_ast_bookkeeping[additional_item_name] = self.__dict__[
                additional_item_name
            ]

    @property
    def has_sys_trace_events(self):
        return any(
            evt in self.events_with_registered_handlers for evt in SYS_TRACE_EVENTS
        )

    @classmethod
    def syntax_augmentation_specs(cls) -> List[AugmentationSpec]:
        specs: List[AugmentationSpec] = []
        for clazz in cls.mro():
            if not issubclass(clazz, BaseTracer):
                continue
            specs.extend(
                spec
                for spec in clazz.__dict__.values()
                if isinstance(spec, AugmentationSpec)
            )
        return specs

    @classmethod
    def get_augmentations(
        cls, node_id: Union[ast.AST, int]
    ) -> FrozenSet[AugmentationSpec]:
        if isinstance(node_id, ast.AST):
            node_id = id(node_id)
        augs = []
        for aug, node_ids in cls.augmented_node_ids_by_spec.items():
            if node_id in node_ids:
                augs.append(aug)
        return frozenset(augs)

    @classmethod
    def make_sandbox_fname(cls) -> str:
        # Increment on the shared base, NOT ``cls``: ``cls.x += 1`` would bind a
        # fresh per-subclass counter, so different tracer subclasses (e.g.
        # pipescript's Macro/Pipeline tracers) would each mint ``<sandbox-1>``,
        # ``<sandbox-2>``, ... and collide on filenames -- and a colliding path
        # makes one module's rewrite evict another's still-live ast bookkeeping.
        _InternalBaseTracer.sandbox_fname_counter += 1
        return f"{SANDBOX_FNAME_PREFIX}-{_InternalBaseTracer.sandbox_fname_counter}>"

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
                        raise exc from None
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
        self,
        path: str,
        module_id: Optional[int] = None,
        tracers: Optional[List["BaseTracer"]] = None,
    ) -> AstRewriter:
        # Only instrument for tracers that aren't hard-disabled right now. A
        # hard-disabled tracer is skipped at event-emit time (see _emit_event), so
        # weaving its events -- and, crucially, its guards -- into the code is pure
        # overhead. This keeps code rewritten while some tracer is disabled lean:
        # e.g. a cooperating tracer that builds lambdas via pyc.eval while *we* are
        # disabled gets a sandbox free of our (unused) guard machinery. When
        # nothing is disabled, pass the live _TRACER_STACK through unchanged so the
        # common path is byte-for-byte identical (no extra list allocation, which
        # some id()-order-sensitive bookkeeping is fragile to).
        #
        # ``tracers`` lets a caller scope the rewrite to a subset of the active
        # stack -- e.g. compiling a sub-fragment that should be instrumented by
        # only some cooperating tracers, not every tracer that happens to be
        # active (a foreign tracer may not recognize the fragment's nodes).
        #
        # ``self`` is always retained even when hard-disabled: a tracer rewrites
        # on its own behalf, and ``instrumented``/``exec`` transiently hard-disable
        # ``self`` only to keep the recompile itself untraced -- ``self`` *will* be
        # live when the rewritten code runs. Dropping it here was a latent bug:
        # with a single tracer the ``or stack`` fallback hid it, but with a second
        # tracer also active, ``self``'s events (e.g. before_call) were silently
        # left out of the woven code.
        stack: List[BaseTracer] = _TRACER_STACK if tracers is None else tracers
        rewrite_tracers: List[BaseTracer] = stack
        if any(tracer._is_tracing_hard_disabled for tracer in stack):
            rewrite_tracers = [
                tracer
                for tracer in stack
                if tracer is self or not tracer._is_tracing_hard_disabled
            ] or stack
        return self.ast_rewriter_cls(rewrite_tracers, path, module_id=module_id)

    def _apply_augmentations(
        self,
        code: str,
        rewriter: Optional[AstRewriter],
        positions: Optional[List[int]],
    ) -> str:
        """Apply this tracer's syntax augmentations to ``code`` -- custom rewrites
        first, then single-token, then paired. When ``positions`` (absolute char
        offsets into ``code``) is given it is remapped *in place* into the returned
        code's coordinates. Shared by the import path (``make_syntax_augmenter``)
        and the eval/exec path (``preprocess``) so custom rewrites behave
        identically on both."""
        aug_specs = self.syntax_augmentation_specs()
        if len(aug_specs) == 0:
            return code
        custom_specs = [spec for spec in aug_specs if spec.is_custom]
        single_specs = [
            spec for spec in aug_specs if not spec.is_custom and not spec.is_paired
        ]
        paired_specs = [
            spec for spec in aug_specs if not spec.is_custom and spec.is_paired
        ]

        if not custom_specs:
            # Legacy fast path: byte-for-byte the previous behavior. Pass the
            # caller's ``positions`` through untouched -- its None-vs-list identity
            # feeds id()-order-sensitive bookkeeping downstream, so never coerce a
            # ``None`` into ``[]`` here.
            code, single_applied = replace_tokens_and_get_augmented_positions(
                code, single_specs, rewriter, positions
            )
            code, paired_applied = (
                replace_paired_delimiters_and_get_augmented_positions(
                    code, paired_specs, rewriter, positions
                )
            )
            if rewriter is not None:
                self.last_applied_specs = single_applied + paired_applied
            return code

        # Custom rewrites run FIRST -- a custom token (e.g. a *leading* ``|>``) may
        # be invalid Python the later passes can't tokenize. Each custom anchor is
        # registered in *post-custom* coordinates (right after its own rewrite,
        # before the single/paired passes), exactly the convention the token/paired
        # passes use for their own registered positions: ``fix_positions`` then
        # shifts every registered position -- custom included -- to its final
        # column for the column-deltas of all *later* specs, whether those run in
        # this same tracer (single/paired below) or in a later tracer on the stack
        # (e.g. pipescript's brace ``[`` registered here, then shifted by a later
        # tracer's ``|>`` -> ``|``). The caller's tracked ``offsets`` are still
        # threaded through every pass via ``remap_through_edits``.
        n_caller = 0 if positions is None else len(positions)
        offsets: List[int] = [] if positions is None else positions
        custom_applied: List[AugmentationSpec] = []

        for spec in custom_specs:
            collected: List[Position] = []
            new_code, edits = spec.custom.rewrite(  # type: ignore[union-attr]
                code, lambda line, col: collected.append(Position(line, col))
            )
            if edits:
                offsets[:] = [remap_through_edits(edits, off) for off in offsets]
            if rewriter is not None:
                for pos in collected:
                    rewriter.register_augmented_position(spec, pos.line, pos.col)
            if collected or edits:
                custom_applied.append(spec)
            code = new_code

        code, single_applied = replace_tokens_and_get_augmented_positions(
            code, single_specs, rewriter, offsets
        )
        code, paired_applied = replace_paired_delimiters_and_get_augmented_positions(
            code, paired_specs, rewriter, offsets
        )

        if rewriter is not None:
            self.last_applied_specs = custom_applied + single_applied + paired_applied

        if positions is not None:
            positions[:] = offsets[:n_caller]
        return code

    def make_syntax_augmenter(
        self, ast_rewriter: Optional[AstRewriter]
    ) -> "Callable[[CodeLines], CodeLines]":
        aug_specs = self.syntax_augmentation_specs()

        def _input_transformer(lines: "CodeLines") -> "CodeLines":
            if len(aug_specs) == 0:
                return lines
            if isinstance(lines, list):
                code = "".join(lines)
            else:
                code = lines
            code = self._apply_augmentations(code, ast_rewriter, None)
            if isinstance(lines, list):
                return code.splitlines(keepends=True)
            else:
                return code

        return _input_transformer

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

    def _augmented_definition_for(self, f: Callable) -> Optional[ast.stmt]:
        """The retained, augmentation-annotated ``def``/``async def`` AST for ``f``.

        When ``f`` was syntax-augmented at compile time (e.g. a notebook-cell helper
        whose body uses a cooperating tracer's surface syntax such as a pipescript
        ``|>``), its augmentations live on the *original* AST nodes -- still reachable
        via ``ast_node_by_id`` -- keyed by node ``id()``. ``inspect.getsource`` returns
        the *lowered* source (the augmented token is already gone), so re-parsing it
        yields fresh nodes with no markings and a cooperating tracer's ``when``-predicated
        handlers never fire. Re-instrumenting from this retained node instead preserves
        them. Returns ``None`` when ``f`` has no retained augmented definition (the
        ordinary case) so callers use the source path.
        """
        code = getattr(f, "__code__", None)
        # Only code woven with pyccolo emits can carry augmentations; a lambda's
        # augmentations live on a ``Lambda`` node, not a named ``def``, so it never
        # matches below. Gating on these keeps the scan off the common (plain) path.
        if (
            code is None
            or code.co_name == "<lambda>"
            or not any(name.endswith("_PYCCOLO_EVT_EMIT") for name in code.co_names)
        ):
            return None
        # Scan the *global* node table, not ``ast_bookkeeper_by_fname[...]``: the latter
        # is rebuilt on every ``instrumented``/``visit`` to hold only that one function's
        # nodes, so a sibling def in the same cell file would already be gone. With
        # ``gc_bookkeeping=False`` every instrumented function's nodes stay live here.
        fallback: Optional[ast.stmt] = None
        for node in self.ast_node_by_id.values():
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == code.co_name
                and any(self.get_augmentations(id(n)) for n in ast.walk(node))
            ):
                # Prefer an exact line match; fall back to a name match (a decorated
                # def's ``co_firstlineno`` can point at the first decorator, not ``def``).
                if getattr(node, "lineno", None) == code.co_firstlineno:
                    return node
                fallback = node
        return fallback

    def instrumented(self, f: Callable, mutate: bool = False) -> Callable:
        f_defined_file = f.__code__.co_filename
        target = f
        with self.tracing_disabled():
            target_name = f.__code__.co_name
            augmented = self._augmented_definition_for(f)
            if augmented is not None:
                # Re-instrument from the retained augmented AST (copied so the live
                # tree is untouched) so syntax augmentations -- e.g. pipescript pipe
                # markings -- survive onto the recompiled node; the lowered linecache
                # source would lose them and degrade pipes to raw operators.
                node: ast.stmt = StatementMapper.bookkeeping_propagating_copy(augmented)
                module = ast.Module(body=[node], type_ignores=[])
            else:
                module = ast.parse(textwrap.dedent(inspect.getsource(f)))
                node = module.body[0]
                if self.instrument_lambdas and not isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    # ``f`` is a lambda: ``getsource`` returns the whole statement that
                    # binds it (``g = lambda ...``, ``foo(lambda ...)``), not a ``def``.
                    # The rewriter only visits module/def nodes, so lift the lambda into
                    # a synthetic ``def`` whose body returns the lambda's expression. Off
                    # by default; a tracer opts in via ``instrument_lambdas = True``.
                    lam = _find_lambda(node, f.__code__.co_argcount)
                    if lam is None:
                        raise ValueError("could not locate lambda in source")
                    target_name = "_pyccolo_lambda"
                    template = ast.parse(f"def {target_name}(): return None").body[0]
                    template.args = lam.args  # type: ignore[attr-defined]
                    template.body = [ast.Return(value=lam.body)]  # type: ignore[attr-defined]
                    ast.copy_location(template, lam)
                    ast.fix_missing_locations(template)
                    node = template
            rewriter = self.make_ast_rewriter(f.__code__.co_filename)
            # ``instrumented`` recompiles a *single* function from a file that may
            # hold other, still-live instrumented code -- most visibly a notebook
            # cell, where one ``co_filename`` is shared by every def/lambda in the
            # cell. The default ``gc_bookkeeping`` assumes a whole-file (re)compile
            # and so evicts the file's prior bookkeeper from the global
            # ``ast_node_by_id`` before re-adding only this function's nodes; that
            # silently drops the bookkeeping of sibling code in the same file (e.g.
            # a pipescript ``|>`` whose runtime handlers then fail their node-id
            # lookups and degrade to raw operators). Add this function's nodes
            # without evicting the rest.
            rewriter.gc_bookkeeping = False
            module.body[0] = rewriter.visit(node)
            # The retained-AST copy carries source locations, but the rewriter injects
            # nodes around them; backfill any the compiler would otherwise reject.
            ast.fix_missing_locations(module)
            compiled: CodeType = compile(module, f.__code__.co_filename, "exec")
            for const in compiled.co_consts:
                if isinstance(const, CodeType) and const.co_name == target_name:
                    if mutate:
                        f.__code__ = const
                    else:
                        target = copy_function_with_code(cast(FunctionType, f), const)
                    break

        @functools.wraps(f)
        def instrumented_f(*args, **kwargs):
            with self.tracing_enabled(tracing_enabled_file=f_defined_file):
                return target(*args, **kwargs)

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
        setattr(builtins, NAME_ERROR_MATCHES, name_error_matches_prefix)
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

    def preprocess(
        self,
        code: str,
        rewriter: Optional[AstRewriter],
        positions: Optional[List[int]] = None,
    ) -> str:
        if len(self.syntax_augmentation_specs()) == 0:
            return code
        if positions is None:
            # Delegate to ``make_syntax_augmenter`` so a subclass override of it
            # (e.g. a cooperating tracer that injects an extra source pass) is
            # honored on the eval/exec path, exactly as before. The base augmenter
            # routes through ``_apply_augmentations``, so custom specs apply here
            # too. Custom-spec support thus lives in one place across both paths.
            return self.make_syntax_augmenter(ast_rewriter=rewriter)(code)
        # Positions-aware path: thread ``positions`` (absolute char offsets,
        # remapped in place) through every pass -- custom, single, then paired --
        # so callers can follow a location across the rewrite.
        return self._apply_augmentations(code, rewriter, positions)

    def parse(
        self,
        code: str,
        mode="exec",
        filename: Optional[str] = None,
        tracers: Optional[List["BaseTracer"]] = None,
        instrument: bool = True,
    ) -> Union[ast.Module, ast.Expression]:
        if filename is None:
            filename = self.make_sandbox_fname()
        rewriter = self.make_ast_rewriter(filename, tracers=tracers)
        for tracer in _TRACER_STACK if tracers is None else tracers:
            code = tracer.preprocess(code, rewriter)
        return rewriter.visit(ast.parse(code, mode=mode), instrument=instrument)

    @overload
    def transform(
        self,
        code: str,
        tracers: Optional[List["BaseTracer"]] = ...,
        positions: None = ...,
        pure: bool = ...,
    ) -> str: ...

    @overload
    def transform(
        self,
        code: str,
        tracers: Optional[List["BaseTracer"]],
        positions: List[Tuple[int, int]],
        pure: bool = ...,
    ) -> Tuple[str, List[Position]]: ...

    @overload
    def transform(
        self,
        code: str,
        *,
        positions: List[Tuple[int, int]],
        pure: bool = ...,
    ) -> Tuple[str, List[Position]]: ...

    def transform(
        self,
        code: str,
        tracers: Optional[List["BaseTracer"]] = None,
        positions: Optional[List[Tuple[int, int]]] = None,
        pure: bool = False,
    ) -> Union[str, Tuple[str, List[Position]]]:
        # ``pure=True`` signals analysis-only use (lint / format / source-map):
        # the result is never executed, so cooperating augmenter callbacks must
        # not perturb execution-relevant state. Setting the flag here covers
        # *both* transform paths -- each iterates the stack calling ``preprocess``
        # -> ``_apply_augmentations`` -> the callbacks, all within this scope.
        # ``preprocess`` itself deliberately takes no ``pure`` param: the flag is
        # already active across these nested calls, and the exec/eval/import paths
        # that also call ``preprocess`` must stay impure.
        token = _PURE_TRANSFORM.set(pure) if pure else None
        try:
            stack = _TRACER_STACK if tracers is None else tracers
            if positions is None:
                for tracer in stack:
                    code = tracer.preprocess(code, rewriter=None)
                return code
            line_starts = _line_starts(code)
            offsets = [offset_of(line_starts, line, col) for line, col in positions]
            for tracer in stack:
                # ``preprocess`` remaps ``offsets`` in place into the coordinates
                # of the code it returns, which is the next tracer's input -- so
                # they compose.
                code = tracer.preprocess(code, rewriter=None, positions=offsets)
            final_starts = _line_starts(code)
            return code, [line_col_of(final_starts, off) for off in offsets]
        finally:
            if token is not None:
                _PURE_TRANSFORM.reset(token)

    def _augmentation_specs_for(
        self, node: ast.AST, tracers: List["BaseTracer"]
    ) -> "FrozenSet[AugmentationSpec]":
        specs: Set[AugmentationSpec] = set()
        for tracer in tracers:
            specs |= tracer.get_augmentations(id(node))
        return frozenset(specs)

    def _reverse_edit_for(
        self,
        rewriter: AstRewriter,
        node: ast.AST,
        spec: AugmentationSpec,
        code: str,
        line_starts: List[int],
    ) -> Optional[Tuple[int, int, str]]:
        """Build a ``(start, end, new_text)`` splice on ``code`` (valid Python) that
        restores ``spec``'s augmented token(s) at ``node``. Returns ``None`` when the
        replacement text can't be located (the node is left untouched)."""
        aug_range = rewriter._range_for_spec(spec, node)
        if aug_range is None:
            return None
        if spec.is_custom:
            return spec.custom.reverse(  # type: ignore[union-attr]
                node, spec, aug_range, code, line_starts
            )
        if spec.is_paired:
            return self._reverse_paired_edit(node, spec, aug_range, code, line_starts)
        start_off = offset_of(line_starts, aug_range.start.line, aug_range.start.col)
        end_off = offset_of(line_starts, aug_range.end.line, aug_range.end.col)
        if spec.aug_type in (AugmentationType.binop, AugmentationType.boolop):
            # The operator lives somewhere in the inter-operand gap; for boolop the
            # range starts at the left value, so begin the search just past its end.
            search_start = start_off
            if spec.aug_type == AugmentationType.boolop:
                end_lineno = getattr(node, "end_lineno", aug_range.start.line)
                end_col = getattr(node, "end_col_offset", aug_range.start.col)
                search_start = offset_of(line_starts, end_lineno, end_col)
            idx = code.find(spec.replacement, search_start, end_off)
            if idx < 0:
                return None
            return (idx, idx + len(spec.replacement), spec.token)
        # prefix / suffix / dot_prefix / dot_suffix / call: the replacement (possibly
        # empty, i.e. a pure deletion) begins exactly at the range anchor.
        if spec.replacement and (
            code[start_off : start_off + len(spec.replacement)] != spec.replacement
        ):
            return None
        return (start_off, start_off + len(spec.replacement), spec.token)

    def _reverse_paired_edit(
        self,
        node: ast.AST,
        spec: AugmentationSpec,
        aug_range: "Range",
        code: str,
        line_starts: List[int],
    ) -> Optional[Tuple[int, int, str]]:
        if not isinstance(node, ast.Subscript):
            return None
        open_off = offset_of(line_starts, aug_range.start.line, aug_range.start.col)
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if end_lineno is None or end_col is None:
            return None
        node_end_off = offset_of(line_starts, end_lineno, end_col)
        close_replacement = (
            spec.close_token
            if spec.close_replacement is None
            else spec.close_replacement
        )
        if spec.body_func_wrapper is not None:
            # The slice is ``wrapper('<inner>', globals(), locals())`` -- recover the
            # original body verbatim from the AST constant rather than the unparsed
            # text (which re-quotes/escapes it). Replace the whole ``[...]`` span.
            sliced: ast.AST = node.slice
            if isinstance(sliced, ast.Index):  # py3.8 compatibility shim
                sliced = sliced.value  # type: ignore[attr-defined]
            if (
                not isinstance(sliced, ast.Call)
                or not isinstance(sliced.func, ast.Name)
                or sliced.func.id != spec.body_func_wrapper
                or len(sliced.args) < 1
                or not isinstance(sliced.args[0], ast.Constant)
                or not isinstance(sliced.args[0].value, str)
            ):
                return None
            inner = sliced.args[0].value
            new_text = spec.token + inner + (spec.close_token or "")
            return (open_off, node_end_off, new_text)
        # Plain paired construct: swap the opening and closing delimiters back. Two
        # disjoint single-delimiter edits keep nested constructs independent.
        if close_replacement is None:
            return None
        if code[open_off : open_off + len(spec.replacement)] != spec.replacement:
            return None
        close_off = node_end_off - len(close_replacement)
        if code[close_off:node_end_off] != close_replacement:
            return None
        # Caller expects a single edit; emit the open swap here and let it also pick
        # up the close swap separately (returned via the dedicated close edit below).
        return (open_off, open_off + len(spec.replacement), spec.token)

    @overload
    def untransform(
        self,
        tree: ast.AST,
        tracers: Optional[List["BaseTracer"]] = ...,
        positions: None = ...,
        pure: bool = ...,
    ) -> str: ...

    @overload
    def untransform(
        self,
        tree: ast.AST,
        tracers: Optional[List["BaseTracer"]],
        positions: List[Tuple[int, int]],
        pure: bool = ...,
    ) -> Tuple[str, List[Position]]: ...

    @overload
    def untransform(
        self,
        tree: ast.AST,
        *,
        positions: List[Tuple[int, int]],
        pure: bool = ...,
    ) -> Tuple[str, List[Position]]: ...

    def untransform(
        self,
        tree: ast.AST,
        tracers: Optional[List["BaseTracer"]] = None,
        positions: Optional[List[Tuple[int, int]]] = None,
        pure: bool = False,
    ) -> Union[str, Tuple[str, List[Position]]]:
        # ``untransform`` invokes ``spec.custom.reverse(...)`` (resugaring); honor
        # the same ``pure`` contract as ``transform`` so a cooperating callback can
        # tell analysis-only resugaring from one tied to execution state. See
        # ``transform`` for why the flag is set here rather than in ``preprocess``.
        token = _PURE_TRANSFORM.set(pure) if pure else None
        try:
            return self._untransform_impl(tree, tracers=tracers, positions=positions)
        finally:
            if token is not None:
                _PURE_TRANSFORM.reset(token)

    def _untransform_impl(
        self,
        tree: ast.AST,
        tracers: Optional[List["BaseTracer"]] = None,
        positions: Optional[List[Tuple[int, int]]] = None,
    ) -> Union[str, Tuple[str, List[Position]]]:
        if sys.version_info < (3, 9):
            raise RuntimeError("untransform requires Python 3.9+ (uses ast.unparse)")
        stack = _TRACER_STACK if tracers is None else tracers
        valid_code = ast.unparse(tree)
        positioned = ast.parse(valid_code)
        line_starts = _line_starts(valid_code)

        # Carry each augmented node's specs from ``tree`` onto the freshly-parsed,
        # correctly-positioned ``positioned`` tree (structurally identical for
        # non-f-string code). Bail out of the walk on the first structural mismatch.
        annotated: List[Tuple[ast.AST, FrozenSet[AugmentationSpec]]] = []
        for orig, pos_node in zip(ast.walk(tree), ast.walk(positioned)):
            if type(orig) is not type(pos_node):
                warnings.warn(
                    "untransform: ast.unparse round-trip diverged structurally "
                    "(likely f-strings); some augmentations may be skipped"
                )
                break
            specs = self._augmentation_specs_for(orig, stack)
            if specs:
                annotated.append((pos_node, specs))

        if not annotated:
            if positions is None:
                return valid_code
            positions_out = [Position(line, col) for line, col in positions]
            return valid_code, positions_out

        filename = self.make_sandbox_fname()
        rewriter = self.make_ast_rewriter(filename, tracers=tracers)
        module_id = id(positioned)
        bookkeeper = AstBookkeeper.create(filename, module_id)
        BookkeepingVisitor(bookkeeper).visit(positioned)
        self.add_bookkeeping(bookkeeper, module_id)
        try:
            edits: List[Tuple[int, int, str]] = []
            for node, specs in annotated:
                for spec in specs:
                    edit = self._reverse_edit_for(
                        rewriter, node, spec, valid_code, line_starts
                    )
                    if edit is not None:
                        edits.append(edit)
                    if spec.is_paired and spec.body_func_wrapper is None:
                        close_edit = self._reverse_paired_close_edit(
                            node, spec, valid_code, line_starts
                        )
                        if close_edit is not None:
                            edits.append(close_edit)
        finally:
            self.remove_bookkeeping(bookkeeper, module_id)

        # Several AST nodes can match the same augmented position (e.g. a ``Name``
        # and the ``Attribute`` containing it), producing duplicate or overlapping
        # edits. Keep one edit per span -- preferring the longest at a given start --
        # and drop any that overlap an already-kept edit. This also leaves ``edits``
        # sorted and non-overlapping, which ``remap_through_edits`` requires.
        edits.sort(key=lambda e: (e[0], -(e[1] - e[0])))
        deduped: List[Tuple[int, int, str]] = []
        last_end = -1
        for start, end, new_text in edits:
            if start < last_end:
                continue
            deduped.append((start, end, new_text))
            last_end = end
        edits = deduped

        # Apply right-to-left so earlier offsets stay valid as we splice.
        out_code = valid_code
        for start, end, new_text in sorted(edits, key=lambda e: e[0], reverse=True):
            out_code = out_code[:start] + new_text + out_code[end:]

        if positions is None:
            return out_code
        length_edits: List[Edit] = sorted(
            (start, end, len(new_text)) for start, end, new_text in edits
        )
        out_starts = _line_starts(out_code)
        positions_out = [
            line_col_of(
                out_starts,
                remap_through_edits(length_edits, offset_of(line_starts, line, col)),
            )
            for line, col in positions
        ]
        return out_code, positions_out

    def _reverse_paired_close_edit(
        self,
        node: ast.AST,
        spec: AugmentationSpec,
        code: str,
        line_starts: List[int],
    ) -> Optional[Tuple[int, int, str]]:
        if not isinstance(node, ast.Subscript):
            return None
        end_lineno = getattr(node, "end_lineno", None)
        end_col = getattr(node, "end_col_offset", None)
        if end_lineno is None or end_col is None or spec.close_token is None:
            return None
        node_end_off = offset_of(line_starts, end_lineno, end_col)
        close_replacement = (
            spec.close_token
            if spec.close_replacement is None
            else spec.close_replacement
        )
        if close_replacement is None:
            return None
        close_off = node_end_off - len(close_replacement)
        if code[close_off:node_end_off] != close_replacement:
            return None
        return (close_off, node_end_off, spec.close_token)

    @contextmanager
    def _preserve_transient_rewrite_state(self) -> Generator[None, None, None]:
        # `parse` overwrites a little transient, non-additive state (the
        # current module and each tracer's last-applied augmentation specs).
        # Save/restore it so a nested parse doesn't clobber an in-flight one.
        saved_module = self.current_module[0]
        saved_specs = [(tracer, tracer.last_applied_specs) for tracer in _TRACER_STACK]
        try:
            yield
        finally:
            self.current_module[0] = saved_module
            for tracer, specs in saved_specs:
                tracer.last_applied_specs = specs

    def parse_fragment(
        self,
        code: str,
        filename: Optional[str] = None,
        tracers: Optional[List["BaseTracer"]] = None,
    ) -> Union[ast.Module, ast.Expression]:
        """Parse + instrument a code fragment from *within* an already active
        trace (e.g. inside an event handler) without disturbing the enclosing
        rewrite's transient state. Returns the instrumented AST; the caller may
        edit it before running it with ``exec_raw(..., instrument=False)``.

        Use this (rather than ``parse``/``exec``) when you need to compile a
        fragment mid-handler: it neither re-enters the tracing context (which
        would reset tracer state) nor clobbers an in-flight parse, so nested
        instrumented constructs in the fragment still dispatch to handlers.

        ``tracers`` scopes the instrumentation to a subset of the active stack;
        pass it when a foreign co-tracer should not weave its events into the
        fragment (e.g. it would not recognize the fragment's synthetic nodes)."""
        if filename is None:
            filename = self.make_sandbox_fname()
        with self._preserve_transient_rewrite_state():
            return self.parse(code, mode="exec", filename=filename, tracers=tracers)

    def exec_fragment(
        self,
        code: Union[str, ast.Module],
        global_env: dict,
        local_env: dict,
        filename: Optional[str] = None,
    ) -> dict:
        """Parse, instrument, and run a code fragment from within an active
        trace, reusing the current tracing context. ``local_env`` is mutated
        with whatever the fragment defines and returned. See
        :meth:`parse_fragment`."""
        if filename is None:
            filename = self.make_sandbox_fname()
        module = (
            self.parse_fragment(code, filename=filename)
            if isinstance(code, str)
            else code
        )
        self.exec_raw(
            module,
            global_env=global_env,
            local_env=local_env,
            filename=filename,
            instrument=False,
        )
        return local_env

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
                code = self.parse(
                    code, mode="eval" if do_eval else "exec", filename=filename
                )
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

    def _register_sandbox_source(
        self, filename: str, source: Optional[Union[str, ast.AST]]
    ) -> None:
        # Make inspect.getsource / tracebacks work for sandbox-compiled code (no
        # on-disk source) by caching its source in linecache. Opt-in, sandbox-only.
        if source is None or not filename.startswith(SANDBOX_FNAME_PREFIX):
            return
        if not (
            self.keep_sandbox_source
            or any(tracer.keep_sandbox_source for tracer in _TRACER_STACK)
        ):
            return
        if not isinstance(source, str):
            if not hasattr(ast, "unparse"):  # ast.unparse is Python 3.9+
                return
            try:
                source = ast.unparse(source)
            except Exception:
                return
        if not isinstance(source, str):
            # Reassigning ``source`` above widens it back to its declared type
            # (``ast.unparse`` is untyped under py3.8 typeshed); re-narrow to str.
            return
        lines = source.splitlines(keepends=True) or [""]
        if not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        linecache.cache[filename] = (len(source), None, lines, filename)

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
        # Capture the source before instrumentation weaves emit calls in: the
        # original string, or the unparsed AST for a synthesized node.
        source: Optional[Union[str, ast.AST]] = code if isinstance(code, str) else None
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
                    code = cast(
                        ast.Expression, self.parse(code, mode="eval", filename=filename)
                    )
                else:
                    code = ast.parse(code, mode="eval", filename=filename)
            if not isinstance(code, ast.Expression):
                code = ast.Expression(code)
            if source is None:
                source = code  # AST input: unparse this (still pre-rewrite below)
            self._register_sandbox_source(filename, source)
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
        code: Union[str, ast.Module, ast.stmt, ast.Expression],
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
        sandboxed_raw_code = textwrap.dedent(
            f"""
            {env_name} = dict(locals())
            def {fun_name}({sandbox_args}):
                return locals()
            {env_name} = {fun_name}(**{env_name})
            {env_name}.pop("__", None)
            {env_name}.pop("builtins", None)
            """
        ).strip("\n")
        sandboxed_code = cast(
            ast.Module, ast.parse(sandboxed_raw_code, filename, "exec")
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
                code = textwrap.dedent(code).strip()
                if instrument:
                    visited = True
                    code = cast(ast.Expression, self.parse(code))
                else:
                    code = ast.parse(code)
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
            lam.__code__ = code.replace(co_name=TRACED_LAMBDA_NAME)
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
                TRACED_LAMBDA_NAME,
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


def make_assert_evt_when(
    orig_when: Optional[Callable[..., bool]],
) -> Callable[..., bool]:
    def when(node: ast.AST) -> bool:
        return isinstance(node, ast.Assert)

    if orig_when is None:
        return when
    else:
        return lambda node: when(node) and orig_when(node)  # type: ignore[misc]


def register_handler(
    event: Union[
        Union[TraceEvent, Type[ast.AST]], Tuple[Union[TraceEvent, Type[ast.AST]], ...]
    ],
    when: Optional[Union[Callable[..., bool], Predicate]] = None,
    reentrant: bool = False,
    static: bool = False,
    use_raw_node_id: bool = False,
    guard: Optional[Callable[[ast.AST], str]] = None,
    exempt_from_guards: bool = False,
):
    events = event if isinstance(event, tuple) else (event,)
    if TraceEvent.before_assert in events or TraceEvent.after_assert in events:
        # TODO: support this
        assert not isinstance(when, Predicate)
        when = make_assert_evt_when(when)
    when = Predicate.TRUE if when is None else when
    if isinstance(when, Predicate):
        pred: Predicate = when.clone()
        pred.static = pred.static or static
        pred.use_raw_node_id = use_raw_node_id
    else:
        pred = Predicate(when, static=static, use_raw_node_id=use_raw_node_id)  # type: ignore

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
                    else EVT_TO_EVENT_MAPPING.get(evt, evt)
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
        when=Predicate.FALSE,
        reentrant=True,
        static=True,
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
