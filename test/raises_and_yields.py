# -*- coding: utf-8 -*-
"""Helper module for ``test_sys_tracer_returns``.

Its functions must live in a real file: ``pyc.exec``'s sandbox filename gets a
special case in the file filter that suppresses the ``call`` event we need.
"""


def boom():
    raise ValueError("boom")


def gen():
    yield 1
    yield 2
