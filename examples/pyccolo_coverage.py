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


logger = logging.getLogger(__name__)


join = os.path.join


EXCEPTED_FILES = {
    'version.py',
    '_version.py',
    # weird shit happens if we instrument _emit_event, so exclude it.
    # can be removed for coverage of non-pyccolo projects.
    'emit_event.py',
}


class CountStatementsVisitor(ast.NodeVisitor):
    def __init__(self):
        self.num_stmts = 0

    def generic_visit(self, node):
        if isinstance(node, ast.stmt):
            self.num_stmts += 1
        super().generic_visit(node)


def count_statements():
    total_by_fname = Counter()
    total = 0
    root = join(os.curdir, pyc.__name__)
    visitor = CountStatementsVisitor()
    for path, _, files in os.walk(root):
        for filename in files:
            if not filename.endswith('.py') or filename in EXCEPTED_FILES:
                continue
            filename = join(path, filename)
            with open(filename, 'r') as f:
                visitor.visit(ast.parse(f.read()))
            total_by_fname[filename] = visitor.num_stmts
            total += visitor.num_stmts
            visitor.num_stmts = 0
    return total, total_by_fname


class CoverageTracer(pyc.BaseTracer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_stmts = set()
        self.stmt_count_by_fname = Counter()

    def file_passes_filter_for_event(self, evt: str, path: str) -> bool:
        if 'test' in path or 'examples' in path:
            # filter out tests and self
            return False

        return 'pyccolo' in path and not any(
            path.endswith(excepted) for excepted in EXCEPTED_FILES
        )

    @pyc.register_raw_handler(ast.stmt)
    def handle_stmt(self, _ret, stmt_id, frame, *_, **__):
        fname = frame.f_code.co_filename
        if fname == "<sandbox>":
            # filter these out. not necessary for non-pyccolo coverage
            return
        if stmt_id not in self.seen_stmts:
            self.stmt_count_by_fname[frame.f_code.co_filename] += 1
            self.seen_stmts.add(stmt_id)


def remove_pyccolo_modules():
    to_delete = []
    for mod in sys.modules:
        if mod.startswith('pyccolo'):
            to_delete.append(mod)
    for mod in to_delete:
        del sys.modules[mod]


if __name__ == "__main__":
    sys.path.insert(0, ".")
    total_stmts, total_by_fname = count_statements()
    # now clear pyccolo modules so that they get reimported, and instrumented
    # can be omitted for non-pyccolo projects
    orig_pyc = pyc
    remove_pyccolo_modules()
    tracer = CoverageTracer.instance()
    with tracer.tracing_context():
        import pyccolo as pyc
        # we just cleared the original tracer stack when we deleted all the imports, so
        # we need to put it back
        # (can be omitted for non-pyccolo projects)
        pyc._TRACER_STACK.append(tracer)

        exit_code = pytest.console_main()
    for fname in sorted(tracer.stmt_count_by_fname.keys()):
        shortened = "." + fname.split(".", 1)[-1]
        seen = tracer.stmt_count_by_fname[fname]
        total = total_by_fname[shortened]
        logger.warning("[%-40s]: seen=%4d, total=%4d, ratio=%.3f", shortened, seen, total, float(seen) / total)
    num_seen_stmts = len(tracer.seen_stmts)
    logger.warning('num stmts seen: %s', num_seen_stmts)
    logger.warning('num stmts total: %s', total_stmts)
    logger.warning('ratio: %.3f', float(num_seen_stmts) / total_stmts)
    sys.exit(exit_code)
