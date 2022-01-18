# -*- coding: utf-8 -*-
"""
Example of simple code coverage implemented using Pyccolo.

Run as `python examples/pyccolo_coverage.py` from the repository root.
"""
import ast
import logging
import os
import sys
from collections import Counter

import pytest

import pyccolo as pyc
from pyccolo.import_hooks import patch_meta_path


logger = logging.getLogger(__name__)


join = os.path.join


EXCEPTED_FILES = {
    "version.py",
    "_version.py",
    # weird shit happens if we instrument _emit_event and import_hooks, so exclude them.
    # can be removed for coverage of non-pyccolo projects.
    "emit_event.py",
    "import_hooks.py",
}


class CountStatementsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.num_stmts = 0

    def generic_visit(self, node):
        if isinstance(node, ast.stmt):
            if not isinstance(node, ast.Raise):
                self.num_stmts += 1
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                return
        super().generic_visit(node)


class CoverageTracer(pyc.BaseTracer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_stmts = set()
        self.stmt_count_by_fname = Counter()
        self.count_static_statements_visitor = CountStatementsVisitor()

    def count_statements(self, path: str) -> int:
        with open(path, "r") as f:
            contents = f.read()
            try:
                self.count_static_statements_visitor.visit(ast.parse(contents))
            except SyntaxError:
                # this means that we must have some other tracer in there,
                # that should be capable of parsing some augmented syntax
                self.count_static_statements_visitor.visit(self.parse(contents))
        ret = self.count_static_statements_visitor.num_stmts
        self.count_static_statements_visitor.num_stmts = 0
        return ret

    def allow_reentrant_events(self) -> bool:
        return False

    def should_instrument_file(self, filename: str) -> bool:
        if "test/" in filename or "examples" in filename:
            # filter out tests and self
            return False

        return "pyccolo" in filename and not any(
            filename.endswith(excepted) for excepted in EXCEPTED_FILES
        )

    @pyc.register_raw_handler(pyc.before_stmt)
    def handle_stmt(self, _ret, stmt_id, frame, *_, **__):
        fname = frame.f_code.co_filename
        if fname == "<sandbox>":
            # filter these out. not necessary for non-pyccolo coverage
            return
        if stmt_id not in self.seen_stmts:
            self.stmt_count_by_fname[fname] += 1
            self.seen_stmts.add(stmt_id)

    def exit_tracing_hook(self) -> None:
        total_stmts = 0
        for fname in sorted(self.stmt_count_by_fname.keys()):
            shortened = "." + fname.split(".", 1)[-1]
            seen = self.stmt_count_by_fname[fname]
            total_in_file = self.count_statements(fname)
            total_stmts += total_in_file
            logger.warning(
                "[%-40s]: seen=%4d, total=%4d, ratio=%.3f",
                shortened,
                seen,
                total_in_file,
                float(seen) / total_in_file,
            )
        num_seen_stmts = len(self.seen_stmts)
        logger.warning("num stmts seen: %s", num_seen_stmts)
        logger.warning("num stmts total: %s", total_stmts)
        logger.warning("ratio: %.3f", float(num_seen_stmts) / total_stmts)


def remove_pyccolo_modules():
    to_delete = []
    for mod in sys.modules:
        if mod.startswith("pyccolo"):
            to_delete.append(mod)
    for mod in to_delete:
        del sys.modules[mod]


if __name__ == "__main__":
    sys.path.insert(0, ".")
    # now clear pyccolo modules so that they get reimported, and instrumented
    # can be omitted for non-pyccolo projects
    orig_pyc = pyc
    remove_pyccolo_modules()
    tracer = CoverageTracer.instance()
    with tracer:
        import pyccolo as pyc

        # we just cleared the original tracer stack when we deleted all the imports, so
        # we need to put it back
        # (can be omitted for non-pyccolo projects)
        pyc._TRACER_STACK.append(tracer)
        with patch_meta_path(pyc._TRACER_STACK):
            exit_code = pytest.console_main()
    sys.exit(exit_code)
