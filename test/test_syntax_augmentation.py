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
