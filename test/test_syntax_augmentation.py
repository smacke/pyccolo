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
            @property
            def syntax_augmentation_specs(self):
                return [add_42_spec]

            @pyc.register_raw_handler(pyc.after_add)
            def handle_add(self, ret, node_id, *_, **__):
                if add_42_spec in self.get_augmentations(node_id):
                    return ret + 42
                else:
                    return ret

        tracer = Add42.instance()
        env = tracer.exec("x = 21 ++ 21")
        assert env["x"] == 84, "got %s" % env["x"]

        env = tracer.exec("x = x + 21", local_env=env)
        assert env["x"] == 105

    coalesce_dot_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.dot, token="?.", replacement="."
    )

    def test_coalescing_dot():
        from pyccolo.examples import NullCoalescer

        NullCoalescer.instance().exec(
            """
            class Foo:
                def __init__(self, x):
                    self.x = x
            foo = Foo(Foo(Foo(None)))
            try:
                bar = foo.x.x.x.x
            except:
                pass
            else:
                assert False
            assert foo.x.x.x?.x is None
            assert foo.x.x.x?.x?.whatever is None
            assert isinstance(foo?.x?.x, Foo)
            assert isinstance(foo.x?.x, Foo)
            assert isinstance(foo?.x.x, Foo)
            """
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

            @property
            def syntax_augmentation_specs(self):
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
