# -*- coding: utf-8 -*-
"""Native IPython / Jupyter integration for Pyccolo.

``%load_ext pyccolo`` installs a *cell tracing driver* on the running shell, and
``pyc.register_ipython_tracer(...)`` adds tracers to it. Instrumentation then
happens on every cell, so there is no need to route code through ``pyc.exec``.

The driver's real interface is :meth:`CellTracingDriver.cell_tracing_context`, a
context manager wrapping the *execution* of one cell. Two hosts drive it:

* **native** -- pyccolo enters the context from ``pre_run_cell`` and exits it from
  ``post_run_cell``.
* **hosted** -- a downstream such as ipyflow, which overrides ``run_cell_async``,
  enters the context directly and supplies the cell name it already computed.

Note that a cell is processed in two phases, and they do *not* nest in a fixed
order. ``InteractiveShell._run_cell`` calls ``transform_cell`` (the source phase)
*before* ``run_cell_async`` triggers ``pre_run_cell`` (the exec phase), whereas a
kernel awaiting ``run_cell_async`` directly gets ``pre_run_cell`` first and
``transform_cell`` second. So the native adapter arms its transformers
persistently rather than per-cell, the source phase builds the rewriter without
assuming tracing is live, and whichever phase runs first opens the cell.

A host takes ownership via :func:`take_over_ipython_driver` and gives it back via
:func:`release_ipython_driver`, so ``%load_ext pyccolo`` and ``%load_ext ipyflow``
compose in either order.

Portability: this module may import from ``IPython.core`` only, and never touches
``shell.kernel``. Both rules keep it working under JupyterLite / Pyodide, where
``ipykernel`` is absent and the shell is a plain ``InteractiveShell``.
"""
import ast
import functools
import os
import re
import sys
from contextlib import ExitStack, contextmanager
from types import FrameType, TracebackType
from typing import (
    TYPE_CHECKING,
    Callable,
    ContextManager,
    Dict,
    Generator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

from pyccolo.ast_rewriter import AstRewriter
from pyccolo.emit_event import SANDBOX_FNAME_PREFIX, is_traceback_visible
from pyccolo.import_hooks import TraceFinder
from pyccolo.trace_events import TraceEvent
from pyccolo.tracer import (
    HIDE_PYCCOLO_FRAME,
    PYCCOLO_DEV_MODE_ENV_VAR,
    TRACED_LAMBDA_NAME,
    BaseTracer,
    register_raw_handler,
)
from pyccolo.utils import resolve_tracer

if TYPE_CHECKING:
    from IPython.core.interactiveshell import InteractiveShell


TracerClass = Type[BaseTracer]
TracerRef = Union[str, TracerClass, BaseTracer]
ExcTuple = Tuple[Type[BaseException], BaseException, Optional[TracebackType]]

# Priority of the tracers a host (e.g. ipyflow) registers on its own behalf.
# Higher sorts later, so user tracers -- registered at the default priority of
# zero -- come first in ``_TRACER_STACK``. This reproduces the ordering ipyflow
# used to get from ``registered_tracers.insert(0, cls)``.
HOST_TRACER_PRIORITY = 100

_LOADED_BY_PYCCOLO = "pyccolo"

_MAGIC_NAME = "pyccolo"

# Marks our ``showtraceback`` wrapper so installs stay idempotent and uninstalls
# never clobber a downstream wrapper layered on top of ours.
_PYCCOLO_SHOWTRACEBACK = "_pyccolo_showtraceback_patch"

_USAGE = "Usage: %pyccolo register|deregister|list [<tracer>]"

# ``test_no_prints`` forbids bare ``print(...)`` in library code so that stray
# debugging output can't ship. The magic's output *is* the feature, so alias it.
print_ = print


# ---------------------------------------------------------------------------
# traceback filtering (shared with downstreams; previously duplicated verbatim
# in pipescript and ipyflow)
# ---------------------------------------------------------------------------


def filter_hidden_frames(tb: Optional[TracebackType]) -> None:
    """Strip pyccolo's own frames out of ``tb``, in place.

    A sandbox frame that was marked traceback-visible via
    :func:`pyccolo.mark_traceback_visible` carries user-authored source (e.g. a
    compiled pipescript block, or a macro sub-lambda) and is kept: it pinpoints
    the failure rather than obscuring it.
    """
    prev: Optional[TracebackType] = None
    while tb is not None:
        should_filter = False
        frame: FrameType = tb.tb_frame
        if prev is not None:
            fname = frame.f_code.co_filename
            should_filter = bool(frame.f_locals.get(HIDE_PYCCOLO_FRAME, False))
            should_filter = should_filter or (
                fname.startswith(SANDBOX_FNAME_PREFIX)
                and not is_traceback_visible(fname)
            )
            should_filter = should_filter or frame.f_code.co_name in (
                TRACED_LAMBDA_NAME,
                "_patched_eval",
                "_patched_tracer_eval",
            )
            should_filter = should_filter or "pyccolo" in fname
        if should_filter and prev is not None:
            prev.tb_next = tb.tb_next
        else:
            prev = tb
        tb = tb.tb_next


# ---------------------------------------------------------------------------
# tracer coercion
# ---------------------------------------------------------------------------


def _adopt_singleton(cls: TracerClass, inst: BaseTracer) -> None:
    if cls.initialized() and cls.instance() is not inst:
        raise ValueError(
            f"{cls.__name__} already has a different singleton instance; "
            "pass the class (or the existing instance) instead"
        )
    # traitlets' ``SingletonConfigurable.instance()`` hands back ``cls._instance``
    # whenever it is set, so this adopts ``inst`` as the singleton.
    cls._instance = inst  # type: ignore[assignment]


def coerce_tracer_class(
    tracer: TracerRef, shell: "Optional[InteractiveShell]" = None
) -> TracerClass:
    """Normalize the four accepted spellings of a tracer to its class.

    A dotted string is imported; a bare string is looked up in the shell's user
    namespace; a class passes through; an instance is adopted as its class's
    singleton, since everything downstream (including ipyflow) stores classes.

    A tracer defined in a cell has ``__module__ == "__main__"``, so
    ``"__main__.MyTracer"`` round-trips. Resolve that spelling against the user
    namespace rather than importing: ``sys.modules["__main__"]`` is the shell's
    user module in a kernel, but is the launching script under an embedded shell.
    """
    resolved: object
    if isinstance(tracer, str):
        name = tracer.strip()
        module_name, _, attr = name.rpartition(".")
        user_ns = {} if shell is None else shell.user_ns
        if not module_name:
            if name not in user_ns:
                raise ValueError(
                    f"no tracer named {name!r} in the user namespace; "
                    "pass a class, an instance, or a fully qualified path"
                )
            resolved = user_ns[name]
        elif module_name == "__main__" and attr in user_ns:
            resolved = user_ns[attr]
        else:
            try:
                resolved = resolve_tracer(name)
            except (AttributeError, ImportError) as exc:
                raise ValueError(f"could not resolve tracer {name!r}: {exc}") from exc
    else:
        resolved = tracer
    if isinstance(resolved, BaseTracer):
        cls = type(resolved)
        _adopt_singleton(cls, resolved)
        return cls
    if isinstance(resolved, type) and issubclass(resolved, BaseTracer):
        return resolved
    raise TypeError(f"expected a pyccolo tracer class or instance; got {resolved!r}")


# ---------------------------------------------------------------------------
# keeping ``Out[N]`` alive
# ---------------------------------------------------------------------------


class _ModuleStmtValueTracer(BaseTracer):
    """Keeps an instrumented cell's trailing expression statement an ``ast.Expr``.

    A ``before_stmt`` handler makes ``StatementInserter`` wrap each statement in
    an ``if`` (see its TODO about saving off expression values). The module body
    then no longer *ends* in an ``ast.Expr``, so IPython's ``last_expr``
    interactivity classifies the cell as non-interactive and silently drops
    ``Out[N]``.

    An ``after_module_stmt`` handler anywhere in the rewrite restores the tail:
    the inserter appends ``EMIT(after_module_stmt, ret=EMIT(_load_saved_expr_stmt_ret))``,
    an ``ast.Expr`` evaluating to the statement's value. ipyflow never hit this
    because its ``DataflowTracer`` registers that event; a bare pyccolo tracer
    does not, so the driver supplies this one.

    It exists only to flip the rewriter's per-event predicate, and so is never
    pushed onto ``_TRACER_STACK``: the value round-trips through the ``after_stmt``
    / ``_load_saved_expr_stmt_ret`` handlers that ``BaseTracer`` already gives
    every tracer.
    """

    instrument_all_files = True
    should_patch_meta_path = False

    @register_raw_handler(TraceEvent.after_module_stmt, reentrant=True)
    def _keep_expr_stmt_value(self, ret: object, *_: object, **__: object) -> object:
        return ret


def _needs_module_stmt_value_tracer(tracers: Sequence[BaseTracer]) -> bool:
    events: Set[TraceEvent] = set()
    for tracer in tracers:
        events |= tracer.events_with_registered_handlers
    return (
        TraceEvent.before_stmt in events and TraceEvent.after_module_stmt not in events
    )


# ---------------------------------------------------------------------------
# per-cell state
# ---------------------------------------------------------------------------


# After IPython's static input transforms, a ``%%foo`` cell body has been folded
# into a string-literal argument to ``run_cell_magic``. It is no longer Python we
# should be augmenting -- token-level rewrites would happily mangle the literal.
_CELL_MAGIC_RE = re.compile(r"^\s*get_ipython\(\)\.run_cell_magic\(")


class CellInfo:
    """State for the cell currently being traced."""

    __slots__ = (
        "raw_cell",
        "transformed_source",
        "cell_name",
        "module_id",
        "tracers",
        "rewrite_tracers",
        "rewriter",
        "syntax_transforms_enabled",
    )

    def __init__(
        self,
        raw_cell: str,
        cell_name: Optional[str],
        module_id: Optional[int],
        tracers: List[BaseTracer],
        syntax_transforms_enabled: bool,
    ) -> None:
        self.raw_cell = raw_cell
        self.transformed_source: Optional[str] = None
        self.cell_name = cell_name
        self.module_id = module_id
        # ``tracers`` are the ones actually pushed onto ``_TRACER_STACK``;
        # ``rewrite_tracers`` may carry an extra one that only shapes the rewrite.
        self.tracers = tracers
        self.rewrite_tracers = list(tracers)
        if _needs_module_stmt_value_tracer(tracers):
            self.rewrite_tracers.append(_ModuleStmtValueTracer.instance())
        self.rewriter: Optional[AstRewriter] = None
        self.syntax_transforms_enabled = syntax_transforms_enabled


class _PyccoloInputTransformer:
    """The single entry pyccolo adds to ``shell.input_transformers_post``.

    One stable object, rather than one closure per tracer, so registering or
    deregistering a tracer never requires surgery on IPython's transformer list.
    """

    def __init__(self, driver: "CellTracingDriver") -> None:
        self._driver = driver

    def __call__(self, lines: List[str]) -> List[str]:
        return self._driver._transform_input(lines)


class _PyccoloAstTransformer(ast.NodeTransformer):
    """The single entry pyccolo adds to ``shell.ast_transformers``."""

    def __init__(self, driver: "CellTracingDriver") -> None:
        self._driver = driver

    def visit(self, node: ast.AST) -> ast.AST:
        return self._driver._transform_ast(node)


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------


class CellTracingDriver:
    """Owns the tracer registry and the per-cell pyccolo lifecycle for one shell."""

    def __init__(self, shell: "InteractiveShell") -> None:
        self.shell = shell

        # (priority, sequence, class), kept sorted; sequence breaks ties so the
        # sort is stable on registration order.
        self._registry: List[Tuple[int, int, TracerClass]] = []
        self._seq = 0
        self._registry_dirty = True

        self._loaded_by: Set[str] = set()
        self._hosted = False

        # Resident tracing: entered once, then merely enabled/disabled per cell.
        self._cleanups: List[Callable[[], None]] = []
        self._active_tracers: List[BaseTracer] = []
        self._saved_meta_path: List[TraceFinder] = []

        self._cell: Optional[CellInfo] = None
        self._in_exec_phase = False
        self._open_ctx: "Optional[ContextManager[Optional[CellInfo]]]" = None
        self._deferred: List[Callable[[], None]] = []

        # --- extension points, set by a host such as ipyflow ---------------
        self.rewriter_factory: Optional[
            Callable[[List[BaseTracer], str, Optional[int]], AstRewriter]
        ] = None
        self.extra_tracers: Optional[Callable[[List[BaseTracer]], List[BaseTracer]]] = (
            None
        )
        self.cell_contexts: List[Callable[[], ContextManager[None]]] = []
        self.on_cell_start: List[Callable[[CellInfo], None]] = []
        self.on_cell_end: List[Callable[[CellInfo], None]] = []
        self.syntax_transforms_enabled = True
        self.syntax_transforms_only = False

        # --- native adapter state -----------------------------------------
        self._native_installed = False
        self._transformers_armed = False
        self._orig_compile_cache: Optional[Callable[..., str]] = None
        self._orig_showtraceback: Optional[Callable[..., None]] = None
        self._orig_transform_cell: Optional[Callable[[str], str]] = None
        self._last_transform: Optional[Tuple[str, str]] = None
        self._input_transformer = _PyccoloInputTransformer(self)
        self._ast_transformer = _PyccoloAstTransformer(self)
        self._magic_installed = False

    # -- registry -----------------------------------------------------------

    def tracers(self) -> List[TracerClass]:
        return [entry[2] for entry in self._registry]

    def register(self, cls: TracerClass, priority: int = 0) -> None:
        # Re-registering replaces, matching ipyflow: a tracer class redefined in
        # a cell is a *different* class object with the same name, and the stale
        # singleton must go.
        self.deregister(cls)
        self._registry.append((priority, self._seq, cls))
        self._seq += 1
        self._registry.sort(key=lambda entry: (entry[0], entry[1]))
        self._registry_dirty = True
        # Deliberately *not* instantiating the singleton here: a tracer may need
        # state that only exists once the host has finished starting up (ipyflow's
        # dataflow tracer wants its flow). ``_resolve_tracers`` builds it lazily,
        # at the first cell where it is actually used.
        if self._is_legacy_ipyflow():
            _legacy_ipyflow_sync(self)

    def deregister(self, cls: TracerClass) -> None:
        # Match on name as well as identity: a tracer class redefined in a cell is
        # a *different* class object, and its stale singleton has to go. Only
        # clear singletons of classes actually leaving the registry -- ``register``
        # routes through here, and clearing an unregistered class's singleton
        # would blow away one the caller is still holding (ipyflow caches its
        # output recorder's instance before registering it).
        stale = [
            entry
            for entry in self._registry
            if entry[2] is cls or entry[2].__name__ == cls.__name__
        ]
        for entry in stale:
            self._registry.remove(entry)
            stale_cls = entry[2]
            if stale_cls.initialized():
                # ``_TRACER_STACK`` is only rebuilt at a cell boundary, so a
                # tracer deregistered *from within a cell* stays resident for the
                # rest of it -- while ``clear_instance`` below drops the singleton
                # its own handlers reach for. Hard-disabling makes ``_emit_event``
                # skip it immediately, so it goes quiet at the moment it is
                # deregistered rather than at the end of the cell.
                stale_cls.instance()._is_tracing_hard_disabled = True
            stale_cls.clear_instance()
        if stale:
            self._registry_dirty = True
            if self._is_legacy_ipyflow():
                _legacy_ipyflow_sync(self)

    def deregister_all(self) -> None:
        for cls in self.tracers():
            self.deregister(cls)

    # -- resident tracing ----------------------------------------------------

    def _tracers_changed(self, tracers: List[BaseTracer]) -> bool:
        """Whether the resident tracers differ from the ones this cell wants.

        Compared by identity, not equality: a re-registered tracer class gets a
        fresh singleton, and a host's ``extra_tracers`` hook may add or drop
        instances from cell to cell (ipyflow toggles its output recorder).
        """
        if len(self._active_tracers) != len(tracers):
            return True
        return any(
            active is not wanted
            for active, wanted in zip(self._active_tracers, tracers)
        )

    def _ensure_tracing(self, tracers: List[BaseTracer]) -> None:
        if self._registry_dirty or self._tracers_changed(tracers):
            self._teardown_tracing()
            self._registry_dirty = False
        if not self._cleanups:
            patch_meta_path = any(tracer.should_patch_meta_path for tracer in tracers)
            last = len(tracers) - 1
            for idx, tracer in enumerate(tracers):
                self._cleanups.append(
                    tracer.tracing_non_context(
                        do_patch_meta_path=patch_meta_path and idx == last
                    )
                )
            self._active_tracers = list(tracers)
        else:
            self._restore_meta_path()
            for tracer in self._active_tracers:
                tracer._enable_tracing(check_disabled=False)

    def _disable_tracing_for_cell(self) -> None:
        for tracer in reversed(self._active_tracers):
            tracer._disable_tracing(check_enabled=False)
        # pyccolo's meta path entries confuse IPython's completer, so park them
        # between cells and put them back before the next one runs.
        while sys.meta_path and isinstance(sys.meta_path[0], TraceFinder):
            self._saved_meta_path.append(cast(TraceFinder, sys.meta_path.pop(0)))

    def _restore_meta_path(self) -> None:
        while self._saved_meta_path:
            sys.meta_path.insert(0, self._saved_meta_path.pop())

    def _teardown_tracing(self) -> None:
        self._restore_meta_path()
        for cleanup in reversed(self._cleanups):
            cleanup()
        self._cleanups.clear()
        self._active_tracers = []

    # -- transformers --------------------------------------------------------

    def _resolve_tracers(self) -> List[BaseTracer]:
        tracers = [cls.instance() for cls in self.tracers()]
        if self.extra_tracers is not None:
            tracers = self.extra_tracers(tracers)
        return tracers

    def _make_rewriter(self, cell: CellInfo) -> AstRewriter:
        path = cell.cell_name or ""
        if self.rewriter_factory is not None:
            return self.rewriter_factory(cell.rewrite_tracers, path, cell.module_id)
        # Scope the rewrite to the cell's tracers explicitly. We cannot fall back
        # on ``_TRACER_STACK`` here: on the ``run_cell`` path the source phase
        # runs before the exec phase has pushed anything onto it.
        return cell.tracers[-1].make_ast_rewriter(
            path, module_id=cell.module_id, tracers=cell.rewrite_tracers
        )

    def _transform_input(self, lines: List[str]) -> List[str]:
        tracers = self._resolve_tracers()
        if not tracers:
            return lines
        source = "".join(lines)
        cell = self._cell
        if cell is None or not self._in_exec_phase:
            # Source phase opened the cell (the ``run_cell`` path), or this is an
            # off-execution ``transform_cell`` -- a completion, or
            # ``should_run_async``. Either way build fresh state: the last source
            # phase before ``pre_run_cell`` is the authoritative one.
            cell = CellInfo(
                raw_cell=source,
                cell_name=None,
                module_id=None,
                tracers=tracers,
                syntax_transforms_enabled=self.syntax_transforms_enabled,
            )
            self._cell = cell
        cell.transformed_source = source
        if cell.rewriter is None:
            cell.rewriter = self._make_rewriter(cell)
        # Otherwise reuse it. A host may call ``transform_cell`` more than once
        # per cell (ipyflow does, to compute the cell name before delegating to
        # ``super().run_cell_async``), and the augmenters register their
        # positions *on the rewriter*. A second pass over already-augmented
        # source finds no tokens, so rebuilding here would silently discard the
        # first pass's positions.
        if not cell.syntax_transforms_enabled or _CELL_MAGIC_RE.match(source):
            return lines
        for tracer in cell.tracers:
            lines = tracer.make_syntax_augmenter(cell.rewriter)(lines)
        return lines

    def _transform_ast(self, node: ast.AST) -> ast.AST:
        cell = self._cell
        if cell is None or cell.rewriter is None or self.syntax_transforms_only:
            return node
        if not cell.cell_name:
            # No filename was bound, so pyccolo's file filter would reject the
            # cell anyway. Leave it uninstrumented rather than guess.
            return node
        return cell.rewriter.visit(node)

    @contextmanager
    def _armed_transformers(self) -> Generator[None, None, None]:
        """Arm the transformers for the duration. Used by hosted mode only.

        The native adapter arms them for the life of the extension instead, since
        the source phase can precede the exec phase.
        """
        shell = self.shell
        old_input = shell.input_transformers_post
        old_ast = shell.ast_transformers
        shell.input_transformers_post = old_input + [self._input_transformer]
        if not self.syntax_transforms_only:
            shell.ast_transformers = old_ast + [self._ast_transformer]
        try:
            yield
        finally:
            shell.input_transformers_post = old_input
            shell.ast_transformers = old_ast

    def _arm_transformers_persistently(self) -> None:
        shell = self.shell
        if self._input_transformer not in shell.input_transformers_post:
            shell.input_transformers_post.append(self._input_transformer)
        if self._ast_transformer not in shell.ast_transformers:
            shell.ast_transformers.append(self._ast_transformer)

    def _disarm_transformers(self) -> None:
        shell = self.shell
        if self._input_transformer in shell.input_transformers_post:
            shell.input_transformers_post.remove(self._input_transformer)
        if self._ast_transformer in shell.ast_transformers:
            shell.ast_transformers.remove(self._ast_transformer)

    def bind_cell_name(self, cell_name: str, module_id: Optional[int] = None) -> None:
        """Tell the driver the filename IPython minted for the current cell.

        pyccolo's file filter admits a file iff it is in the tracer's
        ``_tracing_enabled_files`` (``pyccolo/emit_event.py``), so this is what
        makes the cell instrumentable at all.
        """
        cell = self._cell
        if cell is None:
            return
        cell.cell_name = cell_name
        if module_id is not None:
            cell.module_id = module_id
        if cell.rewriter is not None:
            cell.rewriter._path = cell_name
            if module_id is not None:
                cell.rewriter._module_id = module_id
        for tracer in cell.tracers:
            tracer._tracing_enabled_files.add(cell_name)

    # -- the seam ------------------------------------------------------------

    def _open_exec_cell(
        self,
        raw_cell: str,
        tracers: List[BaseTracer],
        cell_name: Optional[str],
        module_id: Optional[int],
        transformed_cell: Optional[str],
    ) -> CellInfo:
        """Adopt the cell the source phase already opened, or open a fresh one.

        ``transformed_cell`` is IPython's already-transformed source, carried on
        ``ExecutionInfo``. Matching on it -- rather than just taking whatever
        ``self._cell`` holds -- keeps a stale ``CellInfo`` left over from a
        completion's ``transform_cell`` from being mistaken for this cell's.

        Note that it is *presence*, not equality, that decides: our own augmenters
        rewrite the source after we record it, so ``transformed_source`` and
        ``transformed_cell`` legitimately differ for any tracer with a syntax
        augmentation. Comparing them discarded the pending rewriter -- and with it
        every augmented position the source phase had registered -- leaving the
        cell augmented but uninstrumented.
        """
        cell = self._cell
        if cell is None or transformed_cell is None:
            syntax_ok = (
                self.syntax_transforms_enabled and not raw_cell.strip().startswith("%%")
            )
            cell = CellInfo(raw_cell, cell_name, module_id, tracers, syntax_ok)
            self._cell = cell
        else:
            cell.raw_cell = raw_cell
            if cell_name is not None:
                cell.cell_name = cell_name
            if module_id is not None:
                cell.module_id = module_id
        return cell

    @contextmanager
    def cell_tracing_context(
        self,
        raw_cell: str,
        cell_name: Optional[str] = None,
        module_id: Optional[int] = None,
        transformed_cell: Optional[str] = None,
    ) -> Generator[Optional[CellInfo], None, None]:
        """Trace the execution of one cell. Both hosts drive this."""
        self.end_cell()
        tracers = self._resolve_tracers()
        if not tracers:
            # Deregistering the last tracer must still evict the resident ones
            # from ``_TRACER_STACK``; the rebuild in ``_ensure_tracing`` below is
            # never reached to do it for us.
            if self._active_tracers:
                self._teardown_tracing()
            self._registry_dirty = False
            self._cell = None
            yield None
            return
        cell = self._open_exec_cell(
            raw_cell, tracers, cell_name, module_id, transformed_cell
        )
        self._in_exec_phase = True
        try:
            if not self.syntax_transforms_only:
                for tracer in tracers:
                    tracer.reset()
                self._ensure_tracing(tracers)
                if cell.cell_name is not None:
                    self.bind_cell_name(cell.cell_name, cell.module_id)
            with ExitStack() as stack:
                if not self._transformers_armed:
                    stack.enter_context(self._armed_transformers())
                for start_hook in self.on_cell_start:
                    start_hook(cell)
                for make_ctx in self.cell_contexts:
                    stack.enter_context(make_ctx())
                yield cell
        finally:
            try:
                for end_hook in self.on_cell_end:
                    end_hook(cell)
            finally:
                self._close_exec_phase()

    def _close_exec_phase(self) -> None:
        self._cell = None
        self._in_exec_phase = False
        if not self.syntax_transforms_only and self._cleanups:
            self._disable_tracing_for_cell()
        self._drain_deferred()

    def _drain_deferred(self) -> None:
        if self._in_exec_phase or self._open_ctx is not None:
            return
        while self._deferred:
            actions, self._deferred = self._deferred, []
            for action in actions:
                action()

    def end_cell(self) -> None:
        """Exit a cell context left open because ``post_run_cell`` never fired.

        ``pre_run_cell`` is triggered from ``run_cell_async`` but ``post_run_cell``
        from ``run_cell``, and ipykernel re-triggers the latter itself -- so its
        delivery is not something to rely on.
        """
        ctx, self._open_ctx = self._open_ctx, None
        if ctx is not None:
            ctx.__exit__(None, None, None)
        elif self._in_exec_phase:
            self._close_exec_phase()
        self._drain_deferred()

    def _defer_past_current_cell(self, action: Callable[[], None]) -> bool:
        """Postpone reconfiguring the shell until the running cell finishes.

        Every ``%load_ext`` / ``%unload_ext`` executes *as a cell*, so all of this
        would otherwise happen mid-flight:

        * Tearing tracing down drops the ``EMIT_EVENT`` builtin that the rest of
          the already-rewritten cell still calls -- it dies with ``NameError``.
        * Uninstalling unregisters the ``post_run_cell`` handler that has to close
          the cell context still open above us.
        * Installing arms transformers inside a host's per-cell arming, whose
          restore-on-exit then strips them right back out.
        """
        if self._in_exec_phase or self._open_ctx is not None:
            self._deferred.append(action)
            return True
        return False

    # -- native adapter ------------------------------------------------------

    def _pre_run_cell(self, info: object) -> None:
        if self._in_exec_phase and self._open_ctx is None:
            # A host already opened this cell (ipyflow enters the context from
            # ``run_cell_async``, above where ``pre_run_cell`` fires). There is a
            # one-cell window right after ``%unload_ext ipyflow``, before its
            # deferred class-swap eject, where both are live. Let the host drive.
            return
        self.end_cell()
        raw_cell = cast(str, getattr(info, "raw_cell", "") or "")
        ctx = self.cell_tracing_context(
            raw_cell,
            transformed_cell=self._transformed_cell_for(info, raw_cell),
        )
        ctx.__enter__()
        self._open_ctx = ctx

    def _transformed_cell_for(self, info: object, raw_cell: str) -> Optional[str]:
        """The transformed source of the cell ``pre_run_cell`` is announcing.

        ``ExecutionInfo`` only grew a ``transformed_cell`` field in IPython 9.
        Without it ``_open_exec_cell`` discards the pending ``CellInfo``, taking
        the source phase's rewriter -- and every augmented position registered on
        it -- with it, so the cell runs augmented but uninstrumented (``|>``
        degrades to a bare bitwise-or). Fall back to what ``transform_cell``
        recorded, but only when it was this cell's raw source: a completion
        transforms through ``input_transformer_manager`` rather than the shell,
        so a stale entry from some other cell must never be adopted here.
        """
        transformed = cast(Optional[str], getattr(info, "transformed_cell", None))
        if transformed is not None or self._last_transform is None:
            return transformed
        last_raw, last_transformed = self._last_transform
        return last_transformed if last_raw == raw_cell else None

    def _post_run_cell(self, _result: object = None) -> None:
        if self._open_ctx is None:
            # Not ours to close; just let any postponed reconfiguration run.
            self._drain_deferred()
            return
        self.end_cell()

    def _install_native(self) -> None:
        if self._native_installed:
            return
        if self._defer_past_current_cell(self._install_native):
            return
        shell = self.shell
        orig_cache = shell.compile.cache
        self._orig_compile_cache = orig_cache

        # The cell filename must come from the caching compiler that mints it --
        # IPython calls it between ``transform_cell`` and ``transform_ast``.
        # Predicting it from ``execution_count`` breaks on IPython >= 9, which
        # bumps that counter at a different point than IPython 8.
        # (Cells run with ``shell_futures=False`` build a throwaway compiler and
        # so bypass this wrapper; ipykernel and %load_ext always pass True.)
        @functools.wraps(orig_cache)
        def _caching_compiler_cache(
            transformed_code: str, number: int = 0, *args: object, **kwargs: object
        ) -> str:
            cell_name = orig_cache(transformed_code, number, *args, **kwargs)
            self.bind_cell_name(cell_name, number)
            return cell_name

        shell.compile.cache = _caching_compiler_cache  # type: ignore[method-assign]

        # IPython < 9 computes the transformed cell in ``_run_cell`` but never
        # puts it on ``ExecutionInfo``, so ``pre_run_cell`` cannot see it. Record
        # it here; ``_pre_run_cell`` falls back to it, keyed on the raw cell.
        orig_transform_cell = shell.transform_cell
        self._orig_transform_cell = orig_transform_cell

        @functools.wraps(orig_transform_cell)
        def _recording_transform_cell(raw_cell: str) -> str:
            transformed = orig_transform_cell(raw_cell)
            self._last_transform = (raw_cell, transformed)
            return transformed

        shell.transform_cell = _recording_transform_cell  # type: ignore[method-assign]
        shell.events.register("pre_run_cell", self._pre_run_cell)
        shell.events.register("post_run_cell", self._post_run_cell)
        # Armed for the life of the extension, not per cell: ``_run_cell`` calls
        # ``transform_cell`` before ``run_cell_async`` triggers ``pre_run_cell``.
        self._arm_transformers_persistently()
        self._transformers_armed = True
        self._install_showtraceback()
        self._native_installed = True

    def _uninstall_native(self) -> None:
        if not self._native_installed:
            return
        if self._defer_past_current_cell(self._uninstall_native):
            return
        shell = self.shell
        self._teardown_tracing()
        self._disarm_transformers()
        self._transformers_armed = False
        if self._orig_compile_cache is not None:
            shell.compile.cache = self._orig_compile_cache  # type: ignore[method-assign]
            self._orig_compile_cache = None
        if self._orig_transform_cell is not None:
            shell.transform_cell = self._orig_transform_cell  # type: ignore[method-assign]
            self._orig_transform_cell = None
        self._last_transform = None
        for event, handler in (
            ("pre_run_cell", self._pre_run_cell),
            ("post_run_cell", self._post_run_cell),
        ):
            try:
                shell.events.unregister(event, handler)
            except ValueError:
                pass
        self._uninstall_showtraceback()
        self._native_installed = False

    def _install_showtraceback(self) -> None:
        # Patch the *class*, not the instance. Downstreams (pipescript, ipyflow)
        # wrap or override ``showtraceback`` on the class, and an instance
        # attribute would shadow theirs outright -- silently dropping, say,
        # pipescript's traceback re-sugaring. Patching the class instead lets a
        # later wrapper compose over ours.
        shell_cls = type(self.shell)
        orig = cast(Callable[..., None], shell_cls.showtraceback)
        if getattr(orig, _PYCCOLO_SHOWTRACEBACK, False):
            return

        @functools.wraps(orig)
        def patched_showtraceback(
            shell: "InteractiveShell",
            exc_tuple: Optional[ExcTuple] = None,
            *args: object,
            **kwargs: object,
        ) -> None:
            if os.getenv(PYCCOLO_DEV_MODE_ENV_VAR) != "1":
                tb: Optional[TracebackType] = None
                try:
                    _, _, tb = shell._get_exc_info(exc_tuple)
                except ValueError:
                    pass
                filter_hidden_frames(tb)
            orig(shell, exc_tuple, *args, **kwargs)

        setattr(patched_showtraceback, _PYCCOLO_SHOWTRACEBACK, True)
        self._orig_showtraceback = orig
        shell_cls.showtraceback = patched_showtraceback  # type: ignore[assignment]

    def _uninstall_showtraceback(self) -> None:
        orig, self._orig_showtraceback = self._orig_showtraceback, None
        if orig is None:
            return
        shell_cls = type(self.shell)
        # Only restore if ours is still the outermost wrapper; otherwise a
        # downstream wrapped us and unwinding here would drop its patch too.
        if getattr(shell_cls.showtraceback, _PYCCOLO_SHOWTRACEBACK, False):
            shell_cls.showtraceback = orig  # type: ignore[assignment]

    # -- host ownership ------------------------------------------------------

    def _is_legacy_ipyflow(self) -> bool:
        return _legacy_ipyflow_shell(self.shell)

    def sync_host(self) -> None:
        """(Re)establish whichever integration the current host calls for."""
        wants_native = (
            bool(self._loaded_by) and not self._hosted and not self._is_legacy_ipyflow()
        )
        if wants_native:
            self._install_native()
        else:
            self._uninstall_native()

    def shutdown(self) -> None:
        self._remove_magic()
        if self._defer_past_current_cell(self._shutdown_now):
            return
        self._shutdown_now()

    def _shutdown_now(self) -> None:
        self._uninstall_native()
        self._teardown_tracing()
        self._remove_magic()
        _DRIVERS.pop(id(self.shell), None)

    # -- magic ---------------------------------------------------------------

    def _install_magic(self) -> None:
        if self._magic_installed:
            return
        self.shell.register_magics(_make_magics_class())
        self._magic_installed = True

    def _remove_magic(self) -> None:
        if not self._magic_installed:
            return
        magics = self.shell.magics_manager.magics
        magics.get("line", {}).pop(_MAGIC_NAME, None)
        self._magic_installed = False


# ---------------------------------------------------------------------------
# legacy ipyflow interop
#
# pyccolo is upstream of ipyflow, so a released ipyflow that predates the driver
# will still own the cell lifecycle itself. Detect it, install nothing, and push
# registrations through its own registry. Never import ipyflow -- only look for
# an already-imported one.
# ---------------------------------------------------------------------------


def _legacy_ipyflow_shell(shell: "InteractiveShell") -> bool:
    module = sys.modules.get("ipyflow.shell.interactiveshell")
    shell_cls = getattr(module, "IPyflowInteractiveShell", None)
    if shell_cls is None or not isinstance(shell, shell_cls):
        return False
    return not getattr(shell_cls, "uses_pyccolo_driver", False)


def _qualified_name(cls: TracerClass) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def _legacy_ipyflow_sync(driver: CellTracingDriver) -> None:
    """Project ``driver``'s registry onto ipyflow's ``registered_tracers``.

    ipyflow's ``register_tracer`` prepends, so registering in reverse leaves the
    list in pyccolo's registration order -- ahead of ipyflow's own tracers.
    """
    from ipyflow.line_magics import (  # type: ignore[import-not-found]
        deregister_tracer,
        register_tracer,
    )

    shell = driver.shell
    ours = driver.tracers()
    for cls in getattr(shell, "registered_tracers", []):
        if cls in ours:
            deregister_tracer(_qualified_name(cls))
    for cls in reversed(ours):
        register_tracer(_qualified_name(cls), shell_=shell)


# ---------------------------------------------------------------------------
# magic
# ---------------------------------------------------------------------------


def _make_magics_class() -> type:
    from IPython.core.magic import Magics, line_magic, magics_class

    @magics_class
    class PyccoloMagics(Magics):
        # bare ``@line_magic`` names the magic after the method, i.e. ``%pyccolo``
        @line_magic
        def pyccolo(self, line: str) -> None:
            parts = line.strip().split(None, 1)
            cmd = parts[0] if parts else ""
            rest = parts[1].strip() if len(parts) > 1 else ""
            shell = self.shell
            if cmd == "register" and rest:
                register_ipython_tracer(rest, shell=shell)
            elif cmd == "deregister" and rest:
                if rest.lower() == "all":
                    ipython_driver(shell).deregister_all()
                else:
                    deregister_ipython_tracer(rest, shell=shell)
            elif cmd == "list":
                for cls in registered_ipython_tracers(shell=shell):
                    print_(_qualified_name(cls))
            else:
                print_(_USAGE)

    return PyccoloMagics


# ---------------------------------------------------------------------------
# module-level API
# ---------------------------------------------------------------------------


_DRIVERS: Dict[int, CellTracingDriver] = {}


def _get_ipython() -> "InteractiveShell":
    from IPython.core.getipython import get_ipython

    shell = get_ipython()
    if shell is None:
        raise RuntimeError("no active IPython shell")
    return shell


def _driver_for(shell: "Optional[InteractiveShell]") -> CellTracingDriver:
    shell = _get_ipython() if shell is None else shell
    driver = _DRIVERS.get(id(shell))
    if driver is None:
        driver = CellTracingDriver(shell)
        _DRIVERS[id(shell)] = driver
    return driver


def ipython_driver(shell: "Optional[InteractiveShell]" = None) -> CellTracingDriver:
    """The :class:`CellTracingDriver` for ``shell`` (creating it if needed)."""
    return _driver_for(shell)


def _require_loaded(shell: "Optional[InteractiveShell]") -> CellTracingDriver:
    driver = _driver_for(shell)
    if not driver._loaded_by:
        raise RuntimeError(
            "the pyccolo IPython extension is not loaded; run `%load_ext pyccolo` first"
        )
    return driver


def register_ipython_tracer(
    tracer: TracerRef,
    priority: int = 0,
    shell: "Optional[InteractiveShell]" = None,
) -> TracerClass:
    """Instrument every subsequent cell with ``tracer``.

    ``tracer`` may be a :class:`~pyccolo.BaseTracer` subclass, an instance of one,
    a name bound in the user namespace, or a fully qualified ``"pkg.mod.Cls"``.
    Tracers take effect in registration order: the first registered rewrites the
    source outermost and sees events first.
    """
    driver = _require_loaded(shell)
    cls = coerce_tracer_class(tracer, driver.shell)
    driver.register(cls, priority=priority)
    return cls


def deregister_ipython_tracer(
    tracer: TracerRef, shell: "Optional[InteractiveShell]" = None
) -> TracerClass:
    driver = _require_loaded(shell)
    cls = coerce_tracer_class(tracer, driver.shell)
    driver.deregister(cls)
    return cls


def registered_ipython_tracers(
    shell: "Optional[InteractiveShell]" = None,
) -> List[TracerClass]:
    return _driver_for(shell).tracers()


@contextmanager
def cell_tracing_context(
    shell: "InteractiveShell",
    raw_cell: str,
    cell_name: Optional[str] = None,
    module_id: Optional[int] = None,
) -> Generator[Optional[CellInfo], None, None]:
    """Trace one cell of ``shell``. For hosts that drive the lifecycle themselves."""
    with _driver_for(shell).cell_tracing_context(
        raw_cell, cell_name=cell_name, module_id=module_id
    ) as cell:
        yield cell


def end_ipython_cell(shell: "InteractiveShell") -> None:
    """Force-close a cell context, e.g. before tearing a host down."""
    _driver_for(shell).end_cell()


def take_over_ipython_driver(shell: "InteractiveShell") -> CellTracingDriver:
    """Claim the cell lifecycle for a host (e.g. ipyflow).

    pyccolo stops patching the shell; the host is expected to drive
    :meth:`CellTracingDriver.cell_tracing_context` itself.
    """
    driver = _driver_for(shell)
    driver._loaded_by.add("ipyflow")
    driver._hosted = True
    driver.sync_host()
    return driver


def release_ipython_driver(shell: "InteractiveShell") -> None:
    """Give the cell lifecycle back, undoing :func:`take_over_ipython_driver`."""
    driver = _driver_for(shell)
    driver._loaded_by.discard("ipyflow")
    driver._hosted = False
    driver.rewriter_factory = None
    driver.extra_tracers = None
    driver.cell_contexts = []
    driver.on_cell_start = []
    driver.on_cell_end = []
    driver.syntax_transforms_only = False
    if driver._loaded_by:
        driver.sync_host()
    else:
        driver.shutdown()


def load_ipython_extension(shell: "InteractiveShell") -> None:
    driver = _driver_for(shell)
    driver._loaded_by.add(_LOADED_BY_PYCCOLO)
    driver._install_magic()
    driver.sync_host()


def unload_ipython_extension(shell: "InteractiveShell") -> None:
    driver = _DRIVERS.get(id(shell))
    if driver is None:
        return
    driver._loaded_by.discard(_LOADED_BY_PYCCOLO)
    if driver._loaded_by:
        driver._remove_magic()
        driver.sync_host()
    else:
        driver.shutdown()


__all__ = [
    "CellInfo",
    "CellTracingDriver",
    "HOST_TRACER_PRIORITY",
    "cell_tracing_context",
    "coerce_tracer_class",
    "deregister_ipython_tracer",
    "filter_hidden_frames",
    "ipython_driver",
    "load_ipython_extension",
    "register_ipython_tracer",
    "registered_ipython_tracers",
    "release_ipython_driver",
    "take_over_ipython_driver",
    "unload_ipython_extension",
]
