# -*- coding: utf-8 -*-
import sys

from pyccolo.__main__ import make_parser, run

if sys.version_info >= (3, 8):  # noqa

    def test_entrypoint_with_script():
        # just make sure it doesn't raise
        run(
            make_parser().parse_args(
                "./test/uses_optional_chaining.py -t pyccolo.examples.optional_chaining.ScriptOptionalChainer".split()
            )
        )

    def test_entrypoint_with_module():
        # just make sure it doesn't raise
        run(
            make_parser().parse_args(
                "-m test.uses_optional_chaining -t pyccolo.examples.optional_chaining.ScriptOptionalChainer".split()
            )
        )
