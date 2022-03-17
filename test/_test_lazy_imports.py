# -*- coding: utf-8 -*-
from pyccolo.examples import LazyImportTracer


class TestTracer(LazyImportTracer):
    def should_instrument_file(self, filename: str) -> bool:
        return not filename.endswith("lazy_imports.py")


def test_simple():
    with TestTracer.instance():
        import lazy_import_test_module  # noqa: F401


if __name__ == "__main__":
    test_simple()
