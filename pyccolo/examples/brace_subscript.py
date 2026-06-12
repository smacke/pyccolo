# -*- coding: utf-8 -*-
"""
Write subscript-based macros with (multi-line) braces.

Pyccolo's subscript macros -- pyccolo's own :mod:`quick_lambda` (``f[_ + _]``,
``map[...]``) and the pipescript macro suite (``do[...]``, ``fork[...]``,
``repeat[...]``, ...) -- all dispatch on ``node.value.id`` of a ``Subscript``.
This tracer lets you write any of them with braces instead::

    map{ _ + 1 }        # parsed as  map[ _ + 1 ]
    do{
        ...some block...
    }                   # parsed as  do[ ...some block... ]

It does so with a single *paired* :class:`~pyccolo.AugmentationSpec` that swaps
``{`` -> ``[`` and ``}`` -> ``]`` at the source level, correlating the two
delimiters with depth-aware (nesting-safe) matching. The swap is
length-preserving, so the resulting ``Subscript`` is registered as an
augmentation and the original macro handlers fire **unchanged** -- nothing in
quick_lambda / pipescript needs to be rewritten.

Compose it with whatever defines the macros::

    with QuickLambdaTracer:
        with BraceSubscriptTracer:
            assert pyc.eval("f{_ + _}(3, 4)") == 7

By default any non-keyword ``NAME`` immediately followed by ``{`` triggers the
swap (Python keywords like ``return``/``yield`` are excluded because
``return{1}`` is a legal set literal). Pass ``name_pattern`` (a regex) on the
spec to restrict which names are eligible.

Caveat: because ``[...]`` is a subscript *slice*, the brace body must be a valid
expression -- statements (``for``/``if``/assignments) do not parse here. For
genuinely statement-bodied blocks, see
:mod:`pyccolo.examples.block_lambda`.
"""
import pyccolo as pyc


class BraceSubscriptTracer(pyc.BaseTracer):
    # pure source-rewriting tracer; no guard machinery and no handlers of its
    # own -- it just enables the brace syntax for other tracers' subscripts.
    global_guards_enabled = False

    brace_subscript_spec = pyc.AugmentationSpec(
        aug_type=pyc.AugmentationType.subscript,
        token="{",
        replacement="[",
        close_token="}",
        close_replacement="]",
        name_pattern=None,  # any non-keyword NAME; set a regex to restrict
    )
