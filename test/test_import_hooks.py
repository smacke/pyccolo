# -*- coding: utf-8 -*-
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.util import MAGIC_NUMBER, cache_from_source
from pathlib import Path
from typing import Optional

import pytest

import pyccolo as pyc
from pyccolo.import_hooks import TraceLoader

_CACHE_PROBE = """
import ast
import importlib
import json
import os
import runpy
import sys
from pathlib import Path
import pyccolo as pyc

class CacheTracer(pyc.BaseTracer):
    add_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="++", replacement="+"
    )

    @classmethod
    def syntax_augmentation_specs(cls):
        if os.environ.get("CACHE_AUGMENT"):
            return [cls.add_spec]
        return []

    @pyc.after_binop
    def add(self, ret, node, *args, **kwargs):
        return ret + 1 if self.add_spec in self.get_augmentations(node) else ret

    def should_instrument_file(self, filename):
        return filename.endswith("cache_target.py")

    def should_propagate_handler_exception(self, event, exc):
        return True

    @pyc.after_if_test
    def branch(self, ret, node, *args, **kwargs):
        assert isinstance(node, ast.If), repr(node)
        decisions.append(node.lineno)
        return True

decisions = []
error = None
try:
    with CacheTracer.instance():
        for _ in range(int(os.environ.get("CACHE_ATTEMPTS", "1"))):
            try:
                if os.environ.get("CACHE_RUNPY"):
                    runpy.run_module("cache_target", run_name="__main__")
                elif "cache_target" in sys.modules:
                    importlib.reload(sys.modules["cache_target"])
                else:
                    import cache_target
            except RuntimeError as exc:
                error = type(exc).__name__
except BaseException as exc:
    error = type(exc).__name__
print(json.dumps({"decisions": decisions, "error": error,
                  "pyccolo": str(Path(pyc.__file__).resolve())}))
"""


def _run_cache_probe(
    directory: Path, **environment: str
) -> subprocess.CompletedProcess:
    checkout = str(Path(pyc.__file__).resolve().parent.parent)
    env = dict(os.environ, **environment)
    env["PYTHONPATH"] = checkout + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = environment.get("PYTHONDONTWRITEBYTECODE", "")
    env.pop("PYTHONPYCACHEPREFIX", None)
    return subprocess.run(
        [sys.executable, "-c", _CACHE_PROBE],
        cwd=str(directory),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _assert_cache_probe(
    process: subprocess.CompletedProcess,
    error: Optional[str] = None,
    line: int = 1,
    attempts: int = 1,
) -> None:
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert result == {
        "decisions": [line] * attempts,
        "error": error,
        "pyccolo": str(Path(pyc.__file__).resolve()),
    }, (result, process.stderr)


@pytest.mark.parametrize("error", [None, "RuntimeError", "SystemExit"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_cached_import_preserves_nodes_after_initialization_error(
    tmp_path, error, newline
):
    source = "if False:\n    reached = True\n"
    if error is not None:
        source += "raise %s('initialization failed')\n" % error
    source_path = tmp_path / "cache_target.py"
    source_path.write_bytes(source.replace("\n", newline).encode("utf-8"))
    _assert_cache_probe(_run_cache_probe(tmp_path), error)
    cached = next(tmp_path.rglob("cache_target*.pyc"))
    assert (
        int.from_bytes(cached.read_bytes()[12:16], "little")
        == source_path.stat().st_size
    )
    before = cached.stat().st_mtime_ns
    for _ in range(2):
        _assert_cache_probe(_run_cache_probe(tmp_path), error)
        assert cached.stat().st_mtime_ns == before


@pytest.mark.parametrize(
    "source, line",
    [
        (b"\xef\xbb\xbfif False:\n    reached = True\n", 1),
        (b"# coding: latin-1\nif False:\n    reached = 'caf\xe9'\n", 2),
        pytest.param(
            b"if False:\n    reached = True\nassert 1 ++ 1 == 3\n",
            1,
            marks=pytest.mark.skipif(
                sys.version_info < (3, 8),
                reason="Binop augmentation metadata requires AST end positions",
            ),
        ),
    ],
    ids=["utf-8-bom", "latin-1", "syntax-augmentation"],
)
def test_cached_import_uses_original_source_size(tmp_path, source, line):
    (tmp_path / "cache_target.py").write_bytes(source)
    _assert_cache_probe(_run_cache_probe(tmp_path, CACHE_AUGMENT="1"), line=line)
    cached = next(tmp_path.rglob("cache_target*.pyc"))
    assert int.from_bytes(cached.read_bytes()[12:16], "little") == len(source)
    before = cached.stat().st_mtime_ns
    _assert_cache_probe(_run_cache_probe(tmp_path, CACHE_AUGMENT="1"), line=line)
    assert cached.stat().st_mtime_ns == before


def test_loader_preserves_augmented_source_reads(tmp_path):
    class AugmentingTracer(pyc.BaseTracer):
        add_spec = pyc.AugmentationSpec(
            aug_type=pyc.AugmentationType.binop, token="++", replacement="+"
        )

        def should_instrument_file(self, filename: str) -> bool:
            return filename.endswith("cache_target.py")

    source_path = tmp_path / "cache_target.py"
    source_path.write_bytes(b"value = 1 ++ 2\r\n")
    loader = TraceLoader([AugmentingTracer()], "cache_target", str(source_path))
    assert loader.get_data(str(source_path)) == b"value = 1 + 2\n"
    assert loader.get_source("cache_target") == "value = 1 + 2\n"


@pytest.mark.parametrize("flags", [0, 1, 3])
def test_cache_publication_only_corrects_timestamp_source_size(tmp_path, flags):
    class CacheTracer(pyc.BaseTracer):
        requires_ast_bookkeeping = False

        def should_instrument_file(self, filename: str) -> bool:
            return filename.endswith("cache_target.py")

    source_path = tmp_path / "cache_target.py"
    source = b"value = 1\r\n"
    source_path.write_bytes(source)
    loader = TraceLoader([CacheTracer()], "cache_target", str(source_path))
    loader.get_data(str(source_path))
    source_path.write_bytes(source + b"value = 2\r\n")
    cached = Path(cache_from_source(str(source_path)))
    data = MAGIC_NUMBER + flags.to_bytes(4, "little") + b"abcdefgh" + b"bytecode"
    loader.set_data(str(cached), data)
    expected = (
        data[:12] + len(source).to_bytes(4, "little") + data[16:]
        if flags == 0
        else data
    )
    assert cached.read_bytes() == expected


def test_cached_import_survives_process_exit_during_initialization(tmp_path):
    (tmp_path / "cache_target.py").write_text(
        "if False:\n    reached = True\n"
        "import os\nos._exit(int(os.environ['CACHE_EXIT']))\n"
    )
    assert _run_cache_probe(tmp_path, CACHE_EXIT="17").returncode == 17
    cached = next(tmp_path.rglob("cache_target*.pyc"))
    before = cached.stat().st_mtime_ns
    process = _run_cache_probe(tmp_path, CACHE_EXIT="0")
    assert process.returncode == 0, (process.stdout, process.stderr)
    assert process.stdout == "", process.stderr
    assert cached.stat().st_mtime_ns == before


def test_cached_import_refreshes_bookkeeping_when_source_changes(tmp_path):
    source = tmp_path / "cache_target.py"
    source.write_text("if False:\n    reached = True\n")
    _assert_cache_probe(_run_cache_probe(tmp_path))
    source.write_text("\n\nif False:\n    reached = True\n")
    _assert_cache_probe(_run_cache_probe(tmp_path), line=3)
    _assert_cache_probe(_run_cache_probe(tmp_path), line=3)


@pytest.mark.parametrize("error", [None, "RuntimeError"])
def test_cached_import_can_be_retried_or_reloaded(tmp_path, error):
    source = "if False:\n    reached = True\n"
    if error is not None:
        source += "raise RuntimeError('initialization failed')\n"
    (tmp_path / "cache_target.py").write_text(source)
    for _ in range(2):
        _assert_cache_probe(
            _run_cache_probe(tmp_path, CACHE_ATTEMPTS="3"), error, attempts=3
        )


def test_cached_code_preserves_nodes_with_runpy(tmp_path):
    (tmp_path / "cache_target.py").write_text("if False:\n    reached = True\n")
    for _ in range(2):
        _assert_cache_probe(_run_cache_probe(tmp_path, CACHE_RUNPY="1"))


def test_cached_import_rebuilds_incomplete_cache(tmp_path):
    (tmp_path / "cache_target.py").write_text("if False:\n    reached = True\n")
    _assert_cache_probe(_run_cache_probe(tmp_path))
    cached = next(tmp_path.rglob("cache_target*.pyc"))
    cached.write_bytes(cached.read_bytes()[:-10])
    _assert_cache_probe(_run_cache_probe(tmp_path))
    _assert_cache_probe(_run_cache_probe(tmp_path))


def test_cached_import_respects_dont_write_bytecode(tmp_path):
    (tmp_path / "cache_target.py").write_text("if False:\n    reached = True\n")
    for _ in range(2):
        _assert_cache_probe(_run_cache_probe(tmp_path, PYTHONDONTWRITEBYTECODE="1"))
    assert not list(tmp_path.rglob("*.pyc"))
    assert not list(tmp_path.rglob("*.pkl"))


def test_existing_cache_is_read_when_writes_are_disabled(tmp_path):
    (tmp_path / "cache_target.py").write_text("if False:\n    reached = True\n")
    _assert_cache_probe(_run_cache_probe(tmp_path))
    cached = next(tmp_path.rglob("cache_target*.pyc"))
    before = cached.stat().st_mtime_ns
    _assert_cache_probe(_run_cache_probe(tmp_path, PYTHONDONTWRITEBYTECODE="1"))
    assert cached.stat().st_mtime_ns == before


def test_normal_import_does_not_load_instrumented_cache(tmp_path):
    (tmp_path / "cache_target.py").write_text(
        "reached = False\nif False:\n    reached = True\n"
    )
    _assert_cache_probe(_run_cache_probe(tmp_path), line=2)
    process = subprocess.run(
        [sys.executable, "-c", "import cache_target; print(cache_target.reached)"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert process.stdout == "False\n"


def test_concurrent_cache_writers_preserve_nodes(tmp_path):
    (tmp_path / "cache_target.py").write_text("if False:\n    reached = True\n")
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_run_cache_probe, tmp_path) for _ in range(8)]
        for future in futures:
            _assert_cache_probe(future.result())
    _assert_cache_probe(_run_cache_probe(tmp_path))


def test_basic_instrumented_import():
    class IncrementsAssignValue(pyc.BaseTracer):
        def should_instrument_file(self, filename: str) -> bool:
            return filename.endswith("foo.py")

        @pyc.register_handler(pyc.after_assign_rhs)
        def handle_assign(self, ret, node, *_, **__):
            node_id = id(node)
            assert self.ast_node_by_id[node_id] is node
            assert node_id in self.containing_ast_by_id
            assert node_id in self.containing_stmt_by_id
            return ret + 1

    with IncrementsAssignValue.instance().tracing_enabled():
        import test.foo  # noqa
