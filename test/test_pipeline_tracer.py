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
            assert pyc.eval("(1, 2, 3) |> list |> $.index(2)") == 1

    def test_pipeline_methods_nonstandard_whitespace():
        with PipelineTracer:
            assert pyc.eval("(1, 2, 3)   |>     list  |>      $.index(2)") == 1

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
                        "(3, 1, 2) |> (list . reversed . sorted) |> $.index(2).?foo"
                    )
                    is None
                )

    def test_function_placeholder():
        with PipelineTracer:
            with QuickLambdaTracer:
                # assert pyc.eval("(add := (lambda x, y: x + y)) and (add1 := add($, 1)) and add1(42)") == 43
                assert pyc.eval("(add := (lambda x, y: x + y)) and add(42, 1)") == 43
                pyc.exec(
                    "(add := (lambda x, y: x + y)); assert (lambda y: add(42, y))(1)"
                )
                pyc.exec(
                    "(add := (lambda x, y: x + y)); assert (lambda y: add(42, y)) <| 1 == 43"
                )
                pyc.exec("(add := (lambda x, y: x + y)); assert add(42, $) <| 1 == 43")
                pyc.exec("(add := f[$ + $]); assert (add($, 1) <| 1) == 2")
                pyc.exec("(add := f[$ + $]); assert 1 |> add($, 1) == 2")
                pyc.exec("add = f[$ + $]; add1 = add($, 1); assert add1(42) == 43")
                pyc.exec("add = f[$ + $]; assert add($, 42) <| 1 == 43")
                assert pyc.eval("(f[$ + $] |>> add) and add($, 1) <| 1") == 2
                assert pyc.eval("(f[$ + $] |>> add) and 1 |> add($, 1)") == 2

    def test_tuple_unpack_with_placeholders():
        with PipelineTracer:
            with QuickLambdaTracer:
                assert pyc.eval("($, $) *|> $ + $ <|* (1, 2)") == 3
                assert pyc.eval("($, $) *|> $ + $ <|* (1, 2) |> $.real") == 3
                assert pyc.eval("($, $) *|> $ + $ <|* (1, 2) |> $.imag") == 0
                assert pyc.eval("($, $) *|> $ + $ <|* (1, 2) |> $ + 1") == 4
                assert pyc.eval("(1, 2) *|> ($, $) *|> $ + $") == 3

    def test_placeholder_with_kwarg():
        with PipelineTracer:
            pyc.exec("def add(x, y): return x + y; assert 1 |> add($, y=42) == 43")
            pyc.exec("42 |> print($, end=' ')")

    def test_dict_operators():
        with PipelineTracer:
            assert pyc.eval("{'a': 1, 'b': 2} **|> dict") == {"a": 1, "b": 2}
            assert pyc.eval("{'a': 1, 'b': 2} **$> dict <|** {'c': 3, 'd': 4}") == {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
            }
            assert pyc.eval("{'a': 1, 'b': 2} **|> (dict <$** {'c': 3, 'd': 4})") == {
                "a": 1,
                "b": 2,
                "c": 3,
                "d": 4,
            }
            assert pyc.eval("[('a',1), ('b', 2)] |> (list . dict .** dict)") == [
                "a",
                "b",
            ]

    def test_augmentation_spec_order():
        assert PipelineTracer.syntax_augmentation_specs() == [
            PipelineTracer.pipeline_dict_op_spec,
            PipelineTracer.pipeline_tuple_op_spec,
            PipelineTracer.pipeline_op_assign_spec,
            PipelineTracer.pipeline_op_spec,
            PipelineTracer.value_first_left_partial_apply_dict_op_spec,
            PipelineTracer.value_first_left_partial_apply_tuple_op_spec,
            PipelineTracer.value_first_left_partial_apply_op_spec,
            PipelineTracer.function_first_left_partial_apply_dict_op_spec,
            PipelineTracer.function_first_left_partial_apply_tuple_op_spec,
            PipelineTracer.function_first_left_partial_apply_op_spec,
            PipelineTracer.apply_dict_op_spec,
            PipelineTracer.apply_tuple_op_spec,
            PipelineTracer.apply_op_spec,
            PipelineTracer.compose_dict_op_spec,
            PipelineTracer.compose_tuple_op_spec,
            PipelineTracer.compose_op_spec,
            PipelineTracer.arg_placeholder_spec,
        ]
