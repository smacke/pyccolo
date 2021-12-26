# -*- coding: future_annotations -*-
from contextlib import ExitStack, contextmanager


@contextmanager
def multi_context(cms):
    with ExitStack() as stack:
        yield [stack.enter_context(mgr) for mgr in cms]