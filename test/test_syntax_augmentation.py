import sys
import pyccolo as pyc


add_42_spec = pyc.AugmentationSpec(aug_type=pyc.AugmentationType.binop, token='++', replacement='+')


if sys.version_info >= (3, 8):
    def test_augmented_plus():
        class Add42(pyc.BaseTracer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                with self.persistent_fields():
                    self.add_42_node_ids = self.augmented_node_ids_by_spec[add_42_spec]

            @property
            def syntax_augmentation_specs(self):
                return [add_42_spec]

            @pyc.register_raw_handler(pyc.add)
            def handle_add(self, _ret, node_id, *_, left, right, **__):
                ret = left + right
                if node_id in self.add_42_node_ids:
                    return ret + 42
                else:
                    return ret

        tracer = Add42.instance()
        env = tracer.exec("x = 21 ++ 21")
        assert env["x"] == 84, "got %s" % env["x"]

        env = tracer.exec("x = x + 21", local_env=env)
        assert env["x"] == 105
