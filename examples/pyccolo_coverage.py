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


def run_tests():
    exit_code = 0
    try:
        import importlib
        root = join(os.curdir, 'test')
        for path, _, files in os.walk(root):
            for filename in files:
                if not filename.startswith('test') or not filename.endswith('.py'):
                    continue
                mod_name = os.path.splitext(filename)[0]
                module = importlib.import_module(f'test.{mod_name}')
                for attr in dir(module):
                    if attr.startswith('test_'):
                        getattr(module, attr)()
    except Exception:
        exit_code = 1
        logger.exception("exception encountered during tests")
    return exit_code


seen_stmts = set()
stmt_count_by_fname = Counter()


class CoverageTracer(pyc.BaseTracerStateMachine):

    def should_trace_source_path(self, path) -> bool:
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
        if stmt_id not in seen_stmts:
            stmt_count_by_fname[frame.f_code.co_filename] += 1
            seen_stmts.add(stmt_id)


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
        # exit_code = run_tests()
    for fname in sorted(stmt_count_by_fname.keys()):
        shortened = "." + fname.split(".", 1)[-1]
        seen = stmt_count_by_fname[fname]
        total = total_by_fname[shortened]
        logger.warning("[%-40s]: seen=%4d, total=%4d, ratio=%.3f", shortened, seen, total, float(seen) / total)
    num_seen_stmts = len(seen_stmts)
    logger.warning('num stmts seen: %s', num_seen_stmts)
    logger.warning('num stmts total: %s', total_stmts)
    logger.warning('ratio: %.3f', float(num_seen_stmts) / total_stmts)
    sys.exit(exit_code)
