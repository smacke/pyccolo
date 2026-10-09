# -*- coding: utf-8 -*-
import ast
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import SourceFileLoader
from importlib.util import MAGIC_NUMBER, cache_from_source, decode_source, source_hash
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
    command = [sys.executable]
    if "CACHE_HASH_POLICY" in environment:
        command.extend(["--check-hash-based-pycs", environment["CACHE_HASH_POLICY"]])
    return subprocess.run(
        command + ["-c", _CACHE_PROBE],
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
        (
            b"# coding: latin-1\nif False:\n    reached = 'caf\xe9'\n"
            b"assert reached == 'caf\\u00e9'\n",
            2,
        ),
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


@pytest.mark.parametrize("instrumented", [False, True])
@pytest.mark.parametrize(
    "source, decoded",
    [
        (b"value = 1 ++ 2\r\n", "value = 1 ++ 2\n"),
        (b"\xef\xbb\xbfvalue = 1 ++ 2\n", "value = 1 ++ 2\n"),
        (
            b"# coding: latin-1\nvalue = 'caf\xe9'\n",
            "# coding: latin-1\nvalue = 'café'\n",
        ),
    ],
)
def test_loader_separates_raw_and_augmented_source_reads(
    tmp_path, source, decoded, instrumented
):
    class AugmentingTracer(pyc.BaseTracer):
        add_spec = pyc.AugmentationSpec(
            aug_type=pyc.AugmentationType.binop, token="++", replacement="+"
        )

        def should_instrument_file(self, filename: str) -> bool:
            return filename.endswith("cache_target.py")

    source_path = tmp_path / ("cache_target.py" if instrumented else "plain_target.py")
    source_path.write_bytes(source)
    loader = TraceLoader([AugmentingTracer()], "cache_target", str(source_path))
    assert loader.get_data(str(source_path)) == source
    expected = decoded.replace("++", "+") if instrumented else decoded
    assert loader.get_source("cache_target") == expected
    assert loader.get_augmented_source(str(source_path)) == expected


@pytest.mark.parametrize("flags", [0, 1, 3])
def test_cache_publication_preserves_headers(tmp_path, flags):
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
    assert cached.read_bytes() == data


@pytest.fixture
def compilation_loader(tmp_path):
    class CompilationTracer(pyc.BaseTracer):
        requires_ast_bookkeeping = False

        def should_instrument_file(self, filename: str) -> bool:
            return filename.endswith("cache_target.py")

        def make_syntax_augmenter(self, ast_rewriter):
            return lambda source: source.replace("value = 1", "value = 11")

    source_path = tmp_path / "cache_target.py"
    source_path.write_bytes(b"value = 1\r\n")
    return TraceLoader([CompilationTracer()], "cache_target", str(source_path))


@pytest.mark.parametrize("input_type", [bytes, bytearray, memoryview, str, ast.parse])
@pytest.mark.parametrize("instrumented", [False, True])
def test_source_to_code_augments_only_raw_instrumented_inputs(
    compilation_loader, input_type, instrumented
):
    loader = compilation_loader
    prepared = input_type in (str, ast.parse)
    data = input_type("value = 11\n" if prepared else b"value = 1\n")
    path = loader.path if instrumented else "plain_target.py"
    namespace = {}
    exec(loader.source_to_code(data, path), namespace)
    assert namespace["value"] == (11 if instrumented or prepared else 1)


@pytest.mark.parametrize("caching_allowed", [False, True])
def test_compilation_uses_already_read_source(
    compilation_loader, monkeypatch, caching_allowed
):
    loader = compilation_loader
    loader._tracers[0].bytecode_caching_allowed = caching_allowed
    source_path = Path(loader.path)
    original = source_path.read_bytes()
    reads = []
    get_data = SourceFileLoader.get_data

    def read_and_change_source(self, path):
        data = get_data(self, path)
        if path == str(source_path):
            reads.append(data)
            source_path.write_bytes(b"value = 999\n")
        return data

    monkeypatch.setattr(SourceFileLoader, "get_data", read_and_change_source)
    namespace = {}
    exec(loader.get_code("cache_target"), namespace)
    assert namespace["value"] == 11
    assert reads == [original]
    if caching_allowed:
        cached = next(source_path.parent.rglob("*.pyc"))
        assert int.from_bytes(cached.read_bytes()[12:16], "little") == len(original)


@pytest.mark.parametrize("caching_allowed", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, OSError])
def test_augmentation_errors_propagate(compilation_loader, caching_allowed, error_type):
    loader = compilation_loader
    loader._tracers[0].bytecode_caching_allowed = caching_allowed

    def raise_augmentation_error(source):
        raise error_type("augmentation failed")

    loader._syntax_augmenters = [(loader._tracers[0], raise_augmentation_error)]
    source = loader.get_data(loader.path)
    assert source == b"value = 1\r\n"
    inspection_error = ImportError if error_type is OSError else error_type
    with pytest.raises(inspection_error):
        loader.get_source("cache_target")
    with pytest.raises(error_type, match="augmentation failed"):
        loader.source_to_code(source, loader.path)
    with pytest.raises(error_type, match="augmentation failed"):
        loader.get_code("cache_target")
    assert not loader._augmentation_context
    assert not list(Path(loader.path).parent.rglob("*.pyc"))


def test_rewriting_fallback_compiles_augmented_text(compilation_loader, monkeypatch):
    loader = compilation_loader

    def fail_rewriting(node):
        raise ValueError("rewriting failed")

    monkeypatch.setattr(loader._ast_rewriter, "visit", fail_rewriting)
    namespace = {}
    exec(loader.source_to_code(b"value = 1\n", loader.path), namespace)
    assert namespace["value"] == 11


@pytest.mark.parametrize("input_type", [str, ast.parse])
def test_prepared_source_preserves_tracer_selection_fallback(
    compilation_loader, monkeypatch, input_type
):
    loader = compilation_loader

    def fail_selection(path):
        raise ValueError("tracer selection failed")

    monkeypatch.setattr(loader, "get_tracers_for_path", fail_selection)
    namespace = {}
    exec(loader.source_to_code(input_type("value = 11\n"), loader.path), namespace)
    assert namespace["value"] == 11


def test_get_source_wraps_read_errors(compilation_loader):
    loader = compilation_loader
    Path(loader.path).unlink()
    with pytest.raises(ImportError) as raised:
        loader.get_source("cache_target")
    assert raised.value.name == "cache_target"
    assert isinstance(raised.value.__cause__, OSError)


def test_augmenter_can_inspect_source_without_recursing(compilation_loader):
    loader = compilation_loader
    inspected = []

    def augment(source):
        inspected.append(loader.get_source("cache_target"))
        return source.replace("value = 1", "value = 11")

    loader._syntax_augmenters = [(loader._tracers[0], augment)]
    assert loader.get_source("cache_target") == "value = 11\n"
    namespace = {}
    exec(loader.get_code("cache_target"), namespace)
    assert namespace["value"] == 11
    assert inspected == ["value = 1\n", "value = 1\n"]


def test_decoding_can_be_deferred_until_after_augmentation(
    compilation_loader, monkeypatch
):
    loader = compilation_loader
    decoded_inputs = []

    def decode_parseable_source(data):
        decoded_inputs.append(data)
        source = decode_source(data)
        ast.parse(source)
        return source

    monkeypatch.setattr("pyccolo.import_hooks.decode_source", decode_parseable_source)
    loader._syntax_augmenters = [
        (loader._tracers[0], lambda source: source.replace("$", ""))
    ]
    source = b"value = $1\r\n"
    namespace = {}
    exec(loader.source_to_code(source, loader.path), namespace)
    assert namespace["value"] == 1
    assert decoded_inputs == [source, b"value = 1\r\n"]


def test_instrumented_cache_signature_is_versioned(compilation_loader):
    assert "cache-v2" in compilation_loader.make_cache_signature(
        compilation_loader.path
    )
    assert compilation_loader.make_cache_signature("plain_target.py") == "pyccolo"


@pytest.mark.skipif(
    sys.version_info < (3, 8),
    reason="Binop augmentation metadata requires AST end positions",
)
@pytest.mark.parametrize("flags", [1, 3])
@pytest.mark.parametrize("policy", ["default", "always", "never"])
def test_hash_caches_use_original_source_bytes(tmp_path, flags, policy):
    source_path = tmp_path / "cache_target.py"
    source = b"if False:\r\n    reached = True\r\nassert 1 ++ 1 == 3\r\n"
    source_path.write_bytes(source)
    environment = {"CACHE_AUGMENT": "1", "CACHE_HASH_POLICY": policy}
    _assert_cache_probe(_run_cache_probe(tmp_path, **environment))
    cached = next(tmp_path.rglob("cache_target*.pyc"))
    data = cached.read_bytes()
    data = data[:4] + flags.to_bytes(4, "little") + source_hash(source) + data[16:]
    cached.write_bytes(data)
    before = cached.stat().st_mtime_ns
    for writes_disabled in ("", "1"):
        _assert_cache_probe(
            _run_cache_probe(
                tmp_path, PYTHONDONTWRITEBYTECODE=writes_disabled, **environment
            )
        )
        assert cached.read_bytes() == data
        assert cached.stat().st_mtime_ns == before

    original_stat = source_path.stat()
    changed_source = source.replace(b"== 3", b"== 4")
    source_path.write_bytes(changed_source)
    os.utime(source_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    checks_source = policy == "always" or (policy == "default" and flags == 3)
    error = "AssertionError" if checks_source else None
    _assert_cache_probe(_run_cache_probe(tmp_path, **environment), error)
    expected_hash = source_hash(changed_source if checks_source else source)
    assert cached.read_bytes()[4:8] == flags.to_bytes(4, "little")
    assert cached.read_bytes()[8:16] == expected_hash
    after = cached.stat().st_mtime_ns
    _assert_cache_probe(_run_cache_probe(tmp_path, **environment), error)
    assert cached.stat().st_mtime_ns == after


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
