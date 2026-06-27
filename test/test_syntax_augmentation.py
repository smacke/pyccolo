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

    # --- pure (analysis-only) transform ----------------------------------------
    # A custom rewrite modeling pipescript's ``BraceBlockTracer._emit`` hazard:
    # the forward rewrite registers a body into process-global state the runtime
    # later reads -- *unless* the transform is ``pure`` (analysis-only), in which
    # case it emits a valid marker but touches no shared state.
    class _BlockRewrite(pyc.CustomRewrite):
        registry: dict = {}
        counter = 0
        observed: list = []  # records ``is_pure_transform()`` seen per call

        def rewrite(self, code, register):
            from pyccolo.syntax_augmentation import _line_starts, line_col_of

            type(self).observed.append(pyc.is_pure_transform())
            idx = code.find("BLK")
            if idx < 0:
                return code, []
            if pyc.is_pure_transform():
                # analysis only: a valid, lintable marker but no state mutation
                replacement = "blk(0)"
            else:
                type(self).counter += 1
                type(self).registry[type(self).counter] = "body"
                replacement = "blk(%d)" % type(self).counter
            result = code[:idx] + replacement + code[idx + 3 :]
            starts = _line_starts(result)
            pos = line_col_of(starts, idx)  # anchor at the leading ``b``
            register(pos.line, pos.col)
            return result, [(idx, idx + 3, len(replacement))]

        def range_for(self, node):
            return None

        def reverse(self, node, spec, aug_range, code, line_starts):
            return None

    block_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.custom,
        token="BLK",
        replacement="blk",
        custom=_BlockRewrite(),
    )

    class _BlockTracer(pyc.BaseTracer):
        global_guards_enabled = False

        @classmethod
        def syntax_augmentation_specs(cls):
            return [block_spec]

    def _reset_block_state(rw):
        type(rw).registry = {}
        type(rw).counter = 0
        type(rw).observed = []

    def test_pure_transform_leaves_callback_state_untouched():
        tracer = _BlockTracer.instance()
        rw = block_spec.custom
        with tracer:
            # seed shared state as if an execution had registered body #1
            type(rw).counter = 1
            type(rw).registry = {1: "live body"}
            type(rw).observed = []
            before = dict(type(rw).registry)

            out = pyc.transform("x = BLK", pure=True)
            assert out == "x = blk(0)"
            # the callback saw the pure flag, but state is byte-for-byte unchanged
            assert type(rw).observed[-1] is True
            assert type(rw).counter == 1
            assert type(rw).registry == before
            # the flag is reset once the transform returns
            assert pyc.is_pure_transform() is False

    def test_default_transform_mutates_state_backward_compatible():
        tracer = _BlockTracer.instance()
        rw = block_spec.custom
        with tracer:
            _reset_block_state(rw)
            out = pyc.transform("x = BLK")  # pure defaults to False
            assert out == "x = blk(1)"
            assert type(rw).observed[-1] is False
            assert type(rw).counter == 1
            assert type(rw).registry == {1: "body"}

    def test_pure_transform_positions_path_is_also_pure():
        tracer = _BlockTracer.instance()
        rw = block_spec.custom
        with tracer:
            _reset_block_state(rw)
            code = "x = BLK + 1"
            out, positions = pyc.transform(
                code, positions=[(1, code.index("1"))], pure=True
            )
            assert out == "x = blk(0) + 1"
            # the tracked position still lands on the ``1`` after remapping
            line, col = positions[0]
            assert out.splitlines()[line - 1][col] == "1"
            # ... and the positions-aware path touched no shared state either
            assert type(rw).observed[-1] is True
            assert type(rw).counter == 0
            assert type(rw).registry == {}

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

        def test_custom_rewrite_synthesized_node_roundtrip_and_positions():
            # A context-sensitive custom rewrite: a *leading* ``~>`` (one that
            # starts an expression -- here at line start or right after ``=``)
            # becomes ``lambda:``, synthesizing a Lambda the static aug-types can't
            # describe. ``untransform`` strips the synthesized ``lambda:`` back to
            # ``~>``, and tracked positions follow across the rewrite.
            import re

            from pyccolo.syntax_augmentation import Range, offset_of

            class _LeadingArrow(pyc.CustomRewrite):
                _PAT = re.compile(r"(?:^|=\s*)~>")

                def rewrite(self, code, register):
                    spans = [m.end() - 2 for m in self._PAT.finditer(code)]
                    if not spans:
                        return code, []
                    result = code
                    for start in sorted(spans, reverse=True):
                        result = result[:start] + "lambda:" + result[start + 2 :]
                    edits = [(s, s + 2, len("lambda:")) for s in spans]
                    starts = pyc.syntax_augmentation._line_starts(result)
                    delta = 0
                    for s in sorted(spans):
                        pos = pyc.syntax_augmentation.line_col_of(starts, s + delta)
                        register(pos.line, pos.col)
                        delta += len("lambda:") - 2
                    return result, edits

                def range_for(self, node):
                    if isinstance(node, ast.Lambda):
                        return Range.singleton_span(node.lineno, node.col_offset)
                    return None

                def reverse(self, node, spec, aug_range, code, line_starts):
                    if not isinstance(node, ast.Lambda) or node.args.args:
                        return None
                    start = offset_of(line_starts, node.lineno, node.col_offset)
                    body = offset_of(
                        line_starts, node.body.lineno, node.body.col_offset
                    )
                    return (start, body, "~> ")

            arrow_spec = pyc.AugmentationSpec(
                aug_type=pyc.AugmentationType.custom,
                token="~>",
                replacement="lambda:",
                custom=_LeadingArrow(),
            )

            class ArrowTracer(pyc.BaseTracer):
                global_guards_enabled = False

                @classmethod
                def syntax_augmentation_specs(cls):
                    return [arrow_spec]

            tracer = ArrowTracer.instance()
            with tracer:
                # forward transform + eval
                assert tracer.transform("f = ~> 1 + 2") == "f = lambda: 1 + 2"
                assert tracer.eval("~> 1 + 2")() == 3
                # untransform round-trip restores the leading ``~>``
                assert _round_trip(tracer, "f = ~> 1 + 2") == "f = ~> 1 + 2"
                # an infix ``~>`` (not expression-leading) is left untouched
                assert tracer.transform("x = a ~> b") == "x = a ~> b"
                # positions follow across the rewrite: the ``1`` stays on the ``1``
                code = "f = ~> 1 + 2"
                out, positions = tracer.transform(
                    code, positions=[(1, code.index("1"))]
                )
                line, col = positions[0]
                assert out.splitlines()[line - 1][col] == "1"

        def test_custom_rewrite_composes_with_later_spec_positions():
            # A custom spec registered by one pass must have its anchor corrected
            # for column shifts a *later* single-token spec introduces *before* it.
            # ``@@`` -> ``call_`` opens a marked subscript at the ``[``; the later
            # ``++`` -> ``+`` shrinks text to its left, so the ``[`` anchor must be
            # shifted by ``fix_positions`` or it won't bind (and won't reverse).
            from pyccolo.syntax_augmentation import Range

            class _AtRewrite(pyc.CustomRewrite):
                def rewrite(self, code, register):
                    idx = code.find("@@")
                    if idx < 0:
                        return code, []
                    result = code[:idx] + "call_[0]" + code[idx + 2 :]
                    # register the ``[`` anchor in post-custom coordinates
                    starts = pyc.syntax_augmentation._line_starts(result)
                    bracket = idx + len("call_")
                    pos = pyc.syntax_augmentation.line_col_of(starts, bracket)
                    register(pos.line, pos.col)
                    return result, [(idx, idx + 2, len("call_[0]"))]

                def range_for(self, node):
                    if isinstance(node, ast.Subscript):
                        end_l = getattr(node.value, "end_lineno", None)
                        end_c = getattr(node.value, "end_col_offset", None)
                        if end_l is not None and end_c is not None:
                            return Range.singleton_span(end_l, end_c)
                    return None

                def reverse(self, node, spec, aug_range, code, line_starts):
                    from pyccolo.syntax_augmentation import offset_of

                    if not isinstance(node, ast.Subscript):
                        return None
                    # ``@@`` expands to the whole ``call_[0]``, so restore the
                    # entire node span (the brace case instead keeps the macro
                    # name and rewrites only the ``[...]``).
                    o = offset_of(line_starts, node.lineno, node.col_offset)
                    e = offset_of(line_starts, node.end_lineno, node.end_col_offset)
                    return (o, e, "@@")

            at_spec = pyc.AugmentationSpec(
                aug_type=pyc.AugmentationType.custom,
                token="@@",
                replacement="call_[0]",
                custom=_AtRewrite(),
            )
            plus_spec = pyc.AugmentationSpec(
                aug_type=pyc.AugmentationType.binop, token="++", replacement="+"
            )

            class MixedTracer(pyc.BaseTracer):
                global_guards_enabled = False

                @classmethod
                def syntax_augmentation_specs(cls):
                    return [at_spec, plus_spec]

            tracer = MixedTracer.instance()
            with tracer:
                # ``++`` before the ``@@`` shifts the marker's ``[`` left by one;
                # the anchor must still bind so the marker reverses back to ``@@``.
                assert _round_trip(tracer, "r = 1 ++ 2 ++ @@") == "r = 1 ++ 2 ++ @@"
