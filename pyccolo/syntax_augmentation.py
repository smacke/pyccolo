# -*- coding: utf-8 -*-
# flake8: noqa
import re
import sys
from collections import Counter, defaultdict
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Set,
    Tuple,
    Union,
)

if TYPE_CHECKING:
    from pyccolo.ast_rewriter import AstRewriter

    CodeType = Union[str, List[str]]
    if sys.version_info >= (3, 8):
        Pattern = re.Pattern
    else:
        Pattern = Any


class AugmentationType(Enum):
    prefix = "prefix"
    suffix = "suffix"
    dot = "dot"
    binop = "binop"


class AugmentationSpec(NamedTuple):
    aug_type: AugmentationType
    token: str
    replacement: str

    @property
    def escaped_token(self):
        return re.escape(self.token)


AUGMENTED_SYNTAX_REGEX_TEMPLATE = "".join(
    r"^(?:"
    r"   (?:"
    r"      (?!')"
    r"      (?!{q})"
    r"      (?!''')"
    r"      (?!{tq})"
    r"      {any}"
    r"   ) "
    r"   |  {q}[^{q}]*{q}"
    r"   |  '[^']*'"
    r"   |  '''(?:(?!'''){any})*'''"
    r"   |  {tq}(?:(?!{tq}){any})*{tq}"
    r" )*?"
    r" ({{token}})".format(
        q='"',  # quote
        tq='"""',  # triple quote
        any=r"[\S\s]",  # match anything (more general than '.') -- space or non-space
    ).split()
)


def fix_positions(
    pos_by_spec: Dict[AugmentationSpec, Set[Tuple[int, int]]],
    spec_order: Tuple[AugmentationSpec, ...],
) -> Dict[AugmentationSpec, Set[Tuple[int, int]]]:
    grouped_by_line: Dict[int, List[Tuple[int, AugmentationSpec]]] = defaultdict(list)
    fixed_pos_by_spec: Dict[AugmentationSpec, Set[Tuple[int, int]]] = {}
    for spec, positions in pos_by_spec.items():
        fixed_pos_by_spec[spec] = set()
        for line, col in positions:
            grouped_by_line[line].append((col, spec))

    for line, cols_with_spec in grouped_by_line.items():
        total_offset_by_spec: Dict[AugmentationSpec, int] = Counter()
        offset_by_spec: Dict[AugmentationSpec, int] = Counter()
        cols_with_spec.sort()
        for col, spec in cols_with_spec:
            offset = len(spec.token) - len(spec.replacement)
            for prev_applied in spec_order:
                # the offsets will only be messed up for specs that
                # were applied earlier
                total_offset_by_spec[prev_applied] += offset
                if prev_applied == spec:
                    break
            offset_by_spec[spec] += offset
            new_col = col - (total_offset_by_spec[spec] - offset_by_spec[spec])
            fixed_pos_by_spec[spec].add((line, new_col))

    return fixed_pos_by_spec


def replace_tokens_and_get_augmented_positions(
    s: str, spec: AugmentationSpec, regex: "Pattern"
) -> Tuple[str, List[int]]:
    portions = []
    positions = []
    pos_offset = 0
    while True:
        m = regex.match(s)
        if m is None:
            portions.append(s)
            break
        start, _ = m.span(1)
        positions.append(start + pos_offset)
        portions.append(s[:start])
        portions.append(spec.replacement)
        s = s[start + len(spec.token) :]
        pos_offset += start + len(spec.replacement)
    return "".join(portions), positions


def make_syntax_augmenter(
    rewriter: "AstRewriter", aug_spec: AugmentationSpec
) -> "Callable[[CodeType], CodeType]":
    regex = re.compile(
        AUGMENTED_SYNTAX_REGEX_TEMPLATE.format(token=aug_spec.escaped_token)
    )

    def _input_transformer(lines: "CodeType") -> "CodeType":
        if isinstance(lines, list):
            code_lines: List[str] = lines
        else:
            code_lines = lines.splitlines()
        transformed_lines = []
        for idx, line in enumerate(code_lines):
            line, positions = replace_tokens_and_get_augmented_positions(
                line, aug_spec, regex
            )
            transformed_lines.append(line)
            for pos in positions:
                rewriter.register_augmented_position(aug_spec, idx + 1, pos)
        if isinstance(lines, list):
            return transformed_lines
        else:
            return "\n".join(transformed_lines)

    return _input_transformer
