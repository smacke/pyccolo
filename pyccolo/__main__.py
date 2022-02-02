# -*- coding: utf-8 -*-
"""
Allows running scripts and modules with Pyccolo instrumentation enabled.
"""
import argparse
import os
import sys
from pathlib import Path
from runpy import run_module
from typing import List, Type

import pyccolo as pyc


def get_script_as_module(script: str) -> str:
    # ref: https://nvbn.github.io/2016/08/17/ast-import/
    script_path = Path(script)
    script_dir = script_path.parent.as_posix()
    module_name = os.path.splitext(script_path.name)[0]
    sys.path.insert(0, script_dir)
    return module_name


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pyccolo command line tool.")
    parser.add_argument("script", nargs="?", help="Script to run with instrumentation.")
    parser.add_argument(
        "-m", "--module", help="The module to run, if `script` not specified."
    )
    parser.add_argument(
        "-t",
        "--tracer",
        nargs="+",
        help="Tracers to use for instrumentation.",
        required=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.script is None and args.module is None:
        raise ValueError("must specify script, either as file or module")
    if args.script is not None and args.module is not None:
        raise ValueError("only one of `script` or `module` may be specified")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    tracers: List[Type[pyc.BaseTracer]] = []
    for tracer_ref in args.tracer:
        tracers.append(pyc.resolve_tracer(tracer_ref))
    if args.module is not None:
        module_to_run = args.module
    else:
        module_to_run = get_script_as_module(args.script)
    with pyc.multi_context([tracer.instance() for tracer in tracers]):
        run_module(module_to_run)


def main() -> int:
    run(make_parser().parse_args())
    return 0


if __name__ == "__main__":
    sys.exit(main())
