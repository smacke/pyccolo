# -*- coding: utf-8 -*-
import ast
import sys

import pyccolo as pyc

if sys.version_info >= (3, 8):  # noqa
    add_42_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="++", replacement="+"
    )

    def test_augmented_plus():
        class Add42(pyc.BaseTracer):
            @classmethod
            def syntax_augmentation_specs(cls):
                return [add_42_spec]

            @pyc.after_binop(when=lambda node: isinstance(node.op, ast.Add))
            def handle_add(self, ret, node, *_, **__):
                if add_42_spec in self.get_augmentations(id(node)):
                    return ret + 42
                else:
                    return ret

        tracer = Add42.instance()
        env = tracer.exec("x = 21 ++ 21")
        assert env["x"] == 84, "got %s" % env["x"]

        env = tracer.exec("x = x + 21", local_env=env)
        assert env["x"] == 105

    coalesce_dot_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot_suffix, token="?.", replacement="."
    )

    prefix_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.prefix, token="$", replacement=""
    )

    suffix_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.suffix, token="$$", replacement=""
    )

    def test_prefix_suffix():
        class IncrementAugmentedTracer(pyc.BaseTracer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._delayed_increment = 0

            @classmethod
            def syntax_augmentation_specs(cls):
                return [suffix_spec, prefix_spec]

            @pyc.load_name
            def handle_name(self, ret, node, *_, **__):
                self._delayed_increment = 0
                node_id = id(node)
                augs = self.get_augmentations(node_id)
                assert not (prefix_spec in augs and suffix_spec in augs)
                if prefix_spec in augs:
                    offset = 1
                elif suffix_spec in augs:
                    offset = 2
                else:
                    offset = 0
                if isinstance(ret, int):
                    return ret + offset
                else:
                    self._delayed_increment = offset
                    return ret

            @pyc.after_attribute_load
            def handle_attr(self, ret, node, *_, **__):
                if isinstance(node.value, ast.Name):
                    ret += self._delayed_increment
                self._delayed_increment = 0
                augs = self.get_augmentations(id(node))
                assert not (prefix_spec in augs and suffix_spec in augs)
                if prefix_spec in augs:
                    return ret + 1
                elif suffix_spec in augs:
                    return ret + 2
                else:
                    return ret

        with IncrementAugmentedTracer.instance():
            assert (
                pyc.exec(
                    """
                class Foo:
                    y = 4
                x = 3
                foo = Foo()
                z = $x + foo.y$$
                """
                )["z"]
                == 10
            )

            assert (
                pyc.exec(
                    """
                    class Foo:
                        y = 4
                    x = 3
                    foo = Foo()
                    z = $x + $foo.y
                    """
                )["z"]
                == 9
            )

            assert (
                pyc.exec(
                    """
                    class Foo:
                        y = 4
                    x = 3
                    foo = Foo()
                    z = $x + $foo.y$$
                    """
                )["z"]
                == 11
            )

    nullish_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.boolop, token="??", replacement=" or "
    )

    brace_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.subscript,
        token="{",
        replacement="[",
        close_token="}",
        close_replacement="]",
    )

    wrapper_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.subscript,
        token="{",
        replacement="[",
        close_token="}",
        close_replacement="]",
        name_pattern="do",
        body_func_wrapper="__wrap__",
    )

    class _RoundTripTracer(pyc.BaseTracer):
        global_guards_enabled = False

        @classmethod
        def syntax_augmentation_specs(cls):
            return [add_42_spec, coalesce_dot_spec, prefix_spec, nullish_spec]

    def test_transform_positions():
        tracer = _RoundTripTracer.instance()
        with tracer:
            # "x = 21 ++ 21 ++ 3" -> "x = 21 + 21 + 3"
            #  cols 13 ('++') and 16 ('3') shift left by the cumulative -1 each edit
            out, positions = pyc.transform(
                "x = 21 ++ 21 ++ 3", positions=[(1, 0), (1, 5), (1, 13), (1, 16)]
            )
            assert out == "x = 21 + 21 + 3"
            assert positions == [
                pyc.Position(1, 0),  # before any edit: unchanged
                pyc.Position(1, 5),  # before any edit: unchanged
                pyc.Position(1, 12),  # one edit to the left: -1
                pyc.Position(1, 14),  # two edits to the left: -2
            ]

    def test_transform_positions_multiline():
        tracer = _RoundTripTracer.instance()
        with tracer:
            code = "a = 1 ++ 2\nb = a?.real ?? 0\n"
            out, positions = pyc.transform(code, positions=[(1, 9), (2, 4), (2, 15)])
            # boolop's " or " replacement keeps the source's surrounding spaces
            assert out == "a = 1 + 2\nb = a.real  or  0\n"
            # line 1: '2' after '++' shifts -1; line 2 col 4 ('a') unchanged;
            # col 15 ('0') nets +1 ("?." -> "." is -1, "??" -> " or " is +2)
            assert positions == [
                pyc.Position(1, 8),
                pyc.Position(2, 4),
                pyc.Position(2, 16),
            ]

    def test_transform_no_positions_is_backward_compatible():
        tracer = _RoundTripTracer.instance()
        with tracer:
            out = pyc.transform("x = 1 ++ 2")
            assert out == "x = 1 + 2"
            assert isinstance(out, str)

    if sys.version_info >= (3, 9):

        def _round_trip(tracer, code):
            tree = tracer.parse(code, instrument=False)
            return tracer.untransform(tree)

        def test_untransform_roundtrip_single_token():
            tracer = _RoundTripTracer.instance()
            with tracer:
                assert _round_trip(tracer, "x = 21 ++ 21 ++ 3") == "x = 21 ++ 21 ++ 3"
                assert _round_trip(tracer, "y = a?.b?.c") == "y = a?.b?.c"
                assert _round_trip(tracer, "w = $foo") == "w = $foo"
                # boolop normalizes surrounding whitespace but restores the token
                assert _round_trip(tracer, "z = a ?? b") == "z = a??b"

        def test_untransform_roundtrip_paired():
            class BraceTracer(pyc.BaseTracer):
                global_guards_enabled = False

                @classmethod
                def syntax_augmentation_specs(cls):
                    return [brace_spec]

            tracer = BraceTracer.instance()
            with tracer:
                assert _round_trip(tracer, "r = f{_ + _}") == "r = f{_ + _}"
                # nested brace subscripts round-trip too
                assert _round_trip(tracer, "r = map{a{b}}") == "r = map{a{b}}"

        def test_untransform_roundtrip_body_func_wrapper():
            class BlockTracer(pyc.BaseTracer):
                global_guards_enabled = False

                @classmethod
                def syntax_augmentation_specs(cls):
                    return [wrapper_spec]

            tracer = BlockTracer.instance()
            with tracer:
                code = "r = do{ x = 1; y = x + 2 }"
                tree = tracer.parse(code, instrument=False)
                # the wrapper hides the body in a string constant in valid Python
                assert "__wrap__(" in ast.unparse(tree)
                assert tracer.untransform(tree) == code

        def test_untransform_positions():
            tracer = _RoundTripTracer.instance()
            with tracer:
                tree = tracer.parse("y = a?.b?.c", instrument=False)
                # valid form is "y = a.b.c"; col 0 ('y') unchanged, col 8 ('c')
                # shifts right by the two reinserted "?." (one char each before it)
                back, positions = tracer.untransform(tree, positions=[(1, 0), (1, 8)])
                assert back == "y = a?.b?.c"
                assert positions == [pyc.Position(1, 0), pyc.Position(1, 10)]

        def test_untransform_no_augmentations():
            tracer = _RoundTripTracer.instance()
            with tracer:
                tree = tracer.parse("x = 1 + 2", instrument=False)
                assert tracer.untransform(tree) == "x = 1 + 2"

        def test_fstring_augmented_token_does_not_crash():
            # An augmented token inside an f-string expression part must still
            # transform and round-trip without raising (positions may be approximate).
            tracer = _RoundTripTracer.instance()
            with tracer:
                code = 'y = f"{a?.b}"'
                out, positions = pyc.transform(code, positions=[(1, 0)])
                assert out == 'y = f"{a.b}"'
                assert positions == [pyc.Position(1, 0)]
                tree = tracer.parse(code, instrument=False)
                # the "?." is restored (ast.unparse may normalize the quote char)
                assert "a?.b" in tracer.untransform(tree)
