# -*- coding: utf-8 -*-
import sys

import pyccolo as pyc
from pyccolo.examples import PipelineTracer, QuickLambdaTracer

if sys.version_info >= (3, 8):  # noqa

    def test_simple_pipeline():
        with PipelineTracer:
            assert pyc.eval("(1, 2, 3) |> list") == [1, 2, 3]

    def test_simple_pipeline_alt_op():
        with PipelineTracer:
            assert pyc.eval("(1, 2, 3) %>% list") == [1, 2, 3]

    def test_value_first_partial_apply_then_apply():
        with PipelineTracer:
            assert pyc.eval("5 @> isinstance @@ int") is True

    def test_function_first_partial_apply_then_apply():
        with PipelineTracer:
            assert pyc.eval("isinstance <@ 5 @@ int") is True

    def test_pipe_into_value_first_partial_apply():
        with PipelineTracer:
            assert pyc.eval("int |> (5 @> isinstance)") is True

    def test_pipe_into_function_first_partial_apply():
        with PipelineTracer:
            assert pyc.eval("int |> (isinstance <@ 5)") is True

    def test_alt_partial_pipeline_op():
        with PipelineTracer:
            assert pyc.eval("int %>% (5 @> isinstance)") is True

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

    def test_pipeline_methods_nonstandard_whitespace():
        with PipelineTracer:
            assert pyc.eval("(1, 2, 3)   |>     list  |>      .index(2)") == 1

    def test_left_tuple_apply():
        with PipelineTracer:
            assert pyc.eval("(5, int) *|> isinstance") is True

    def test_right_tuple_apply():
        with PipelineTracer:
            assert pyc.eval("isinstance <|* (5, int)") is True
