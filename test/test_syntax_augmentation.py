import ast
import sys
import pyccolo as pyc


if sys.version_info >= (3, 8):

    add_42_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.binop, token="++", replacement="+"
    )

    def test_augmented_plus():
        class Add42(pyc.BaseTracer):
            @property
            def syntax_augmentation_specs(self):
                return [add_42_spec]

            @pyc.register_raw_handler(pyc.add)
            def handle_add(self, _ret, node_id, *_, left, right, **__):
                ret = left + right
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
        class CoalescingDotTracer(pyc.BaseTracer):
            class DotIsAlwaysNone:
                def __getattr__(self, _item):
                    return None

            dot_is_always_none = DotIsAlwaysNone()

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                with self.persistent_fields():
                    self.coalesce_dot_ids = self.augmented_node_ids_by_spec[
                        coalesce_dot_spec
                    ]

            @property
            def syntax_augmentation_specs(self):
                return [coalesce_dot_spec]

            @pyc.register_raw_handler(ast.Attribute)
            def handle_attr_dot(self, ret, node_id, *_, **__):
                if ret is None and node_id in self.coalesce_dot_ids:
                    return self.dot_is_always_none
                else:
                    return ret

        CoalescingDotTracer.instance().exec(
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
