# -*- coding: utf-8 -*-
import ast
import importlib.util
import logging
import sys
import threading
from contextlib import contextmanager
from importlib.abc import MetaPathFinder
from importlib.machinery import SourceFileLoader
from importlib.util import decode_source, find_spec, spec_from_loader
from types import CodeType, ModuleType
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple

from pyccolo.trace_events import TraceEvent
from pyccolo.utils import clone_function

if TYPE_CHECKING:
    from pyccolo.tracer import BaseTracer

logger = logging.getLogger(__name__)


orig_cache_from_source = clone_function(importlib.util.cache_from_source)  # type: ignore
orig_source_from_cache = clone_function(importlib.util.source_from_cache)  # type: ignore


_pyccolo_loader: "TraceLoader" = None  # type: ignore


def pyccolo_cache_from_source(path, debug_override=None, *, optimization=None):
    path, ext = path.rsplit(".", 1)
    return orig_cache_from_source(
        f"{path}.{_pyccolo_loader.make_cache_signature(path)}.{ext}",
        debug_override=debug_override,
        optimization=optimization,
    )


def pyccolo_source_from_cache(path):
    parts = path.split(".")
    if len(parts) >= 3 and parts[-3].startswith("pyccolo"):
        return orig_source_from_cache(".".join(parts[:-3] + parts[-2:]))
    else:
        return orig_source_from_cache(path)


class TraceLoader(SourceFileLoader):
    def __init__(self, tracers: List["BaseTracer"], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tracers = tracers
        self._ast_rewriter = tracers[-1].make_ast_rewriter()
        self._syntax_augmenters: List[Tuple["BaseTracer", List[Callable]]] = []
        for tracer in tracers:
            self._syntax_augmenters.append(
                (tracer, tracer.make_syntax_augmenters(self._ast_rewriter))
            )
        self._augmentation_context: bool = False

    def make_cache_signature(self, path: str) -> str:
        version_dict: Dict[str, str] = {}
        suffix_parts = []
        for tracer in self._tracers:
            if tracer.should_instrument_file(path):
                tracer_cls = tracer.__class__
                suffix_parts.append(tracer_cls.__name__)
                pkg = tracer_cls.__module__.split(".")[0]
                pkg_version = getattr(sys.modules.get(pkg), "__version__", None)
                if isinstance(pkg_version, (int, str)):
                    version_dict[pkg] = str(pkg_version)
        return "-".join(
            ["pyccolo"]
            + sorted("-".join(v) for v in version_dict.items())
            + suffix_parts
        ).replace(".", "_")

    @contextmanager
    def syntax_augmentation_context(self):
        orig_aug_context = self._augmentation_context
        try:
            self._augmentation_context = True
            yield
        finally:
            self._augmentation_context = orig_aug_context

    @contextmanager
    def patch_cache_handlers(self) -> Generator[None, None, None]:
        cfs_globals = importlib.util.cache_from_source.__globals__
        sfc_globals = importlib.util.source_from_cache.__globals__
        try:
            importlib.util.cache_from_source.__code__ = (
                pyccolo_cache_from_source.__code__
            )
            importlib.util.source_from_cache.__code__ = (
                pyccolo_source_from_cache.__code__
            )
            cfs_globals["orig_cache_from_source"] = orig_cache_from_source
            sfc_globals["orig_source_from_cache"] = orig_source_from_cache
            cfs_globals["_pyccolo_loader"] = sfc_globals["_pyccolo_loader"] = self
            yield
        finally:
            importlib.util.cache_from_source.__code__ = orig_cache_from_source.__code__
            importlib.util.source_from_cache.__code__ = orig_source_from_cache.__code__
            cfs_globals.pop("orig_cache_from_source", None)
            sfc_globals.pop("orig_source_from_cache", None)
            cfs_globals.pop("_pyccolo_loader", None)
            sfc_globals.pop("_pyccolo_loader", None)

    def get_data(self, path: str) -> bytes:
        parts = path.split(".")
        if parts[-1] == "pyc":
            if len(parts) >= 3 and parts[-3] == self.make_cache_signature(path):
                return super().get_data(path)
            path = orig_source_from_cache(path)
        path_str = str(path)
        if self._augmentation_context or not any(
            tracer._should_instrument_file_impl(path_str) for tracer in self._tracers
        ):
            return super().get_data(path)
        with self.syntax_augmentation_context():
            source = self.get_augmented_source(path)
            return bytes(source, encoding="utf-8")

    def get_filename(self, name: Optional[str] = None) -> str:
        source_path = super().get_filename(name)
        for tracer in reversed(self._tracers):
            source_path = tracer._emit_event(
                TraceEvent.before_import.value,
                None,
                sys._getframe(),
                ret=source_path,
                qualified_module_name=name,
            )
        return source_path

    def get_code(self, fullname) -> Optional[CodeType]:
        with self.patch_cache_handlers():
            return super().get_code(fullname)

    def get_augmented_source(self, source_path) -> str:
        source_bytes = super().get_data(source_path)
        still_needs_decode = True
        try:
            source = decode_source(source_bytes)
            still_needs_decode = False
        except SyntaxError:
            # This allows us to handle esoteric encodings that
            # require parsing, such as future_annotations.
            # In this case, just guess that it's utf-8 encoded.

            # This is a bit unfortunate in that it involves multiple
            # round-trips of decoding / encoding, but it's the only
            # way I can think of to ensure that source transformations
            # happen in the correct order.
            source = str(source_bytes, encoding="utf-8")
        source_path_str = str(source_path)
        for tracer, augmenters in self._syntax_augmenters:
            if not tracer._should_instrument_file_impl(source_path_str):
                continue
            for augmenter in augmenters:
                source = augmenter(source)
        if still_needs_decode:
            source = decode_source(bytes(source, encoding="utf-8"))
        return source

    def source_to_code(self, data, path, *, _optimize=-1) -> CodeType:  # type: ignore[override]
        path_str = str(path)
        try:
            if any(
                tracer._should_instrument_file_impl(path_str)
                for tracer in self._tracers
            ):
                return compile(
                    self._ast_rewriter.visit(ast.parse(data)),
                    path,
                    "exec",
                    dont_inherit=True,
                    optimize=_optimize,
                )
            else:
                return super().source_to_code(data, path, _optimize=_optimize)  # type: ignore[call-arg]
        except Exception:
            logger.exception("exception during source to code for path %s", path)
            return super().source_to_code(data, path, _optimize=_optimize)  # type: ignore[call-arg]

    def exec_module(self, module: ModuleType) -> None:
        source_path = str(self.get_filename(module.__name__))
        should_reenable_saved_state = []
        for tracer in reversed(self._tracers):
            should_disable = (
                tracer._is_tracing_enabled
                and not tracer._should_instrument_file_impl(source_path)
            )
            should_reenable_saved_state.append(should_disable)
            if should_disable:
                tracer._disable_tracing()
        should_reenable_saved_state.reverse()
        super().exec_module(module)
        for tracer, should_reenable in zip(self._tracers, should_reenable_saved_state):
            tracer._emit_event(
                TraceEvent.after_import.value, None, sys._getframe(), module=module
            )
            if should_reenable:
                tracer._enable_tracing()


# this is based on the birdseye finder (which uses import hooks based on MacroPy's):
# https://github.com/alexmojaki/birdseye/blob/9974af715b1801f9dd99fef93ff133d0ab5223af/birdseye/import_hook.py
class TraceFinder(MetaPathFinder):
    def __init__(self, tracers) -> None:
        self.tracers = tracers
        self._thread = threading.current_thread()

    @contextmanager
    def _clear_preceding_finders(self) -> Generator[None, None, None]:
        """
        Clear all preceding finders from sys.meta_path, and restore them afterwards.
        """
        orig_finders = sys.meta_path
        try:
            sys.meta_path = sys.meta_path[sys.meta_path.index(self) + 1 :]  # noqa: E203
            yield
        finally:
            sys.meta_path = orig_finders

    def _find_plain_spec(self, fullname, path, target):
        """Try to find the original module using all the
        remaining meta_path finders."""
        spec = None
        self_seen = False
        for finder in sys.meta_path:
            if finder is self:
                self_seen = True
                continue
            elif not self_seen or "pytest" in finder.__module__:
                # when testing with pytest, it installs a finder that for
                # some yet unknown reasons makes birdseye
                # fail. For now it will just avoid using it and pass to
                # the next one
                continue
            if hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, target=target)
            elif hasattr(finder, "load_module"):
                spec = spec_from_loader(fullname, finder)

            if spec is not None and spec.origin != "builtin":
                return spec

    def find_spec(self, fullname, path=None, target=None):
        if threading.current_thread() is not self._thread:
            return None
        if target is None:
            with self._clear_preceding_finders():
                spec = find_spec(fullname, path)
        else:
            spec = self._find_plain_spec(fullname, path, target)
        if spec is None or not (
            hasattr(spec.loader, "get_source") and callable(spec.loader.get_source)
        ):  # noqa: E128
            if fullname != "org":
                # stdlib pickle.py at line 94 contains a ``from
                # org.python.core for Jython which is always failing,
                # of course
                logger.debug("Failed finding spec for %s", fullname)
            return None

        if not isinstance(spec.loader, SourceFileLoader):
            return None
        source_path = spec.loader.get_filename(fullname)
        tracers_to_use = []
        for tracer in self.tracers:
            if (
                tracer._should_instrument_file_impl(source_path)
                or tracer._file_passes_filter_impl(
                    TraceEvent.before_import.value, source_path
                )
                or tracer._file_passes_filter_impl(
                    TraceEvent.after_import.value, source_path
                )
            ):
                tracers_to_use.append(tracer)
        if len(tracers_to_use) == 0:
            return None
        spec.loader = TraceLoader(tracers_to_use, spec.loader.name, spec.loader.path)
        return spec


def patch_meta_path_non_context(tracers: List["BaseTracer"]) -> Callable:
    orig_meta_path_entry = None

    def cleanup_callback():
        if orig_meta_path_entry is None:
            del sys.meta_path[0]
        else:
            sys.meta_path[0] = orig_meta_path_entry

    if len(sys.meta_path) > 0 and isinstance(sys.meta_path[0], TraceFinder):
        orig_meta_path_entry = sys.meta_path[0]
        sys.meta_path[0] = TraceFinder(tracers)
    else:
        sys.meta_path.insert(0, TraceFinder(tracers))
    return cleanup_callback


@contextmanager
def patch_meta_path(tracers: List["BaseTracer"]) -> Generator[None, None, None]:
    cleanup_callback = None
    try:
        cleanup_callback = patch_meta_path_non_context(tracers)
        yield
    finally:
        if cleanup_callback is not None:
            cleanup_callback()
