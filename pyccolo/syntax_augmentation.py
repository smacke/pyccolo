# -*- coding: utf-8 -*-
import itertools
from collections import Counter, defaultdict
from enum import Enum
from io import StringIO
from tokenize import TokenInfo
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Set, Tuple, Union

from IPython.core.inputtransformer2 import make_tokens_by_line

if TYPE_CHECKING:
    from pyccolo.ast_rewriter import AstRewriter

    CodeType = Union[str, List[str]]


class AugmentationType(Enum):
    prefix = "prefix"
    suffix = "suffix"
    dot = "dot"
    binop = "binop"


class AugmentationSpec(NamedTuple):
    aug_type: AugmentationType
    token: str
    replacement: str


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
    tokenizable: Union[str, List[TokenInfo]], spec: AugmentationSpec
) -> Tuple[str, List[Tuple[int, int]]]:
    if isinstance(tokenizable, str):
        tokens = list(make_tokens_by_line([tokenizable]))[0]
    else:
        tokens = tokenizable
    transformed = StringIO()
    match = StringIO()
    cur_match_start = (-1, -1)

    def _flush_match(force: bool = False) -> None:
        if force or not spec.token.startswith(match.getvalue()):
            transformed.write(match.getvalue())
            match.seek(0)
            match.truncate()

    def _write_match(tok: TokenInfo) -> None:
        nonlocal cur_match_start
        if match.getvalue() == "":
            cur_match_start = tok.start
        match.write(tok.string)

    idx = 0
    col_offset = 0
    positions = []
    prev = None
    while idx < len(tokens):
        cur = tokens[idx]
        if prev is not None and prev.end[0] == cur.start[0]:
            match.write(" " * (cur.start[1] - prev.end[1]))
            _flush_match()
        else:
            col_offset = 0
            _flush_match(force=True)
            match.write(" " * cur.start[1])
        _write_match(cur)
        _flush_match()
        if spec.token == match.getvalue():
            positions.append((cur_match_start[0], cur_match_start[1] + col_offset))
            col_offset += len(spec.replacement) - len(spec.token)
            transformed.write(spec.replacement)
            match.seek(0)
            match.truncate()
        prev = cur
        idx += 1

    _flush_match(force=True)
    return transformed.getvalue(), positions


def make_syntax_augmenter(
    rewriter: "AstRewriter", aug_spec: AugmentationSpec
) -> "Callable[[CodeType], CodeType]":
    def _input_transformer(lines: "CodeType") -> "CodeType":
        if isinstance(lines, list):
            code_lines: List[str] = lines
        else:
            code_lines = lines.splitlines(keepends=True)
        tokens = list(itertools.chain(*make_tokens_by_line(code_lines)))
        transformed, positions = replace_tokens_and_get_augmented_positions(
            tokens, aug_spec
        )
        for pos in positions:
            rewriter.register_augmented_position(aug_spec, *pos)
        if isinstance(lines, list):
            return transformed.splitlines(keepends=True)
        else:
            return transformed

    return _input_transformer
