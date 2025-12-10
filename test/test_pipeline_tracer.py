# -*- coding: utf-8 -*-
import sys

import pyccolo as pyc
from pyccolo.examples import OptionalChainer, PipelineTracer, QuickLambdaTracer

if sys.version_info >= (3, 8):  # noqa

    def test_simple_pipeline():
        with PipelineTracer:
            assert pyc.eval("(1, 2, 3) |> list") == [1, 2, 3]

    def test_value_first_partial_apply_then_apply():
        with PipelineTracer:
            assert pyc.eval("5 $> isinstance <| int") is True

    def test_fake_infix():
        with PipelineTracer:
            assert pyc.eval("5 $>isinstance<| int") is True

    def test_value_first_partial_tuple_apply_then_apply():
        with PipelineTracer:
            assert pyc.eval("(1, 2) *$> (lambda a, b, c: a + b + c) <| 3") == 6

    def test_value_first_partial_tuple_apply_then_apply_quick_lambda():
        with PipelineTracer:
            with QuickLambdaTracer:
                assert pyc.eval("(1, 2) *$> f[_ + _ + _] <| 3") == 6

    def test_function_first_partial_apply_then_apply():
        with PipelineTracer:
            assert pyc.eval("isinstance <$ 5 <| int") is True

    def test_function_first_partial_tuple_apply_then_apply():
        with PipelineTracer:
            assert pyc.eval("(lambda a, b, c: a + b + c) <$* (1, 2) <| 3") == 6

    def test_function_first_partial_tuple_apply_then_apply_quick_lambda():
        with PipelineTracer:
            with QuickLambdaTracer:
                assert pyc.eval("f[_ + _ + _] <$* (1, 2) <| 3") == 6

    def test_pipe_into_value_first_partial_apply():
        with PipelineTracer:
            assert pyc.eval("int |> (5 $> isinstance)") is True

    def test_pipe_into_function_first_partial_apply():
        with PipelineTracer:
            assert pyc.eval("int |> (isinstance <$ 5)") is True

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

    def test_compose_op():
        with PipelineTracer:
            assert pyc.eval("((lambda x: x * 5) . (lambda x: x + 2))(10)") == 60

    def test_tuple_compose_op():
        with PipelineTracer:
            assert (
                pyc.eval("((lambda x, y: x * 5 + y) .* (lambda x: (x, x + 2)))(10)")
                == 62
            )

    def test_compose_op_no_space():
        with PipelineTracer:
            assert pyc.eval("((lambda x: x * 5). (lambda x: x + 2))(10)") == 60

    def test_compose_op_extra_space():
        with PipelineTracer:
            assert pyc.eval("((lambda x: x * 5)  . (lambda x: x + 2))(10)") == 60

    def test_compose_op_with_parenthesized_quick_lambdas():
        with PipelineTracer:
            with QuickLambdaTracer:
                assert pyc.eval("((f[_ * 5]) . (f[_ + 2]))(10)") == 60

    def test_compose_op_with_quick_lambdas():
        with PipelineTracer:
            with QuickLambdaTracer:
                assert pyc.eval("(f[_ * 5] . f[_ + 2])(10)") == 60

    def test_pipeline_inside_quick_lambda():
        with PipelineTracer:
            with QuickLambdaTracer:
                assert pyc.eval("2 |> f[_ |> f[_ + 2]]") == 4

    def test_pipeline_dot_op_with_optional_chain():
        with PipelineTracer:
            with OptionalChainer:
                assert (
                    pyc.eval(
                        "(3, 1, 2) |> (list . reversed . sorted) |> .index(2).?foo"
                    )
                    is None
                )
