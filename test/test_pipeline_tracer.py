# -*- coding: utf-8 -*-
import sys

import pyccolo as pyc
from pyccolo.examples import PipelineTracer, QuickLambdaTracer

if sys.version_info >= (3, 8):  # noqa

    def test_simple_pipeline():
        with PipelineTracer:
            assert pyc.eval("(1, 2, 3) |> list") == [1, 2, 3]

    def test_simple_pipeline_with_quick_lambda_map():
        with PipelineTracer:
            with QuickLambdaTracer:
                assert pyc.eval("(1, 2, 3) |> f[map(f[_ + 1], _)] |> list") == [2, 3, 4]

    def test_pipeline_assignment():
        with PipelineTracer:
            with QuickLambdaTracer:
                assert pyc.eval(
                    "(1, 2, 3) |> list |>> result |> f[map(f[_ + 1], _)] |> list |> f[result + _]"
                ) == [1, 2, 3, 2, 3, 4]

    def test_pipeline_methods():
        with PipelineTracer:
            assert pyc.eval("(1, 2, 3) |> list |> .index(2)") == 1
