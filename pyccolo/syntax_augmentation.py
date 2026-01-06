# -*- coding: utf-8 -*-
import itertools
import tokenize
import warnings
from enum import Enum
from io import StringIO
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Set, Tuple, Union

if TYPE_CHECKING:
    from pyccolo.ast_rewriter import AstRewriter

    CodeType = Union[str, List[str]]


class AugmentationType(Enum):
    prefix = "prefix"
    suffix = "suffix"
    dot_prefix = "dot_prefix"
    dot_suffix = "dot_suffix"
    binop = "binop"
    boolop = "boolop"
    call = "call"


class AugmentationSpec(NamedTuple):
    aug_type: AugmentationType
    token: str
    replacement: str


def fix_positions(
    pos_by_spec: Dict[AugmentationSpec, Set[Tuple[int, int]]],
    spec_order: Tuple[AugmentationSpec, ...],
) -> Dict[AugmentationSpec, Set[Tuple[int, int]]]:
    col_by_spec_by_line: Dict[int, Dict[AugmentationSpec, List[int]]] = {}
    fixed_pos_by_spec: Dict[AugmentationSpec, Set[Tuple[int, int]]] = {}
    for spec, positions in pos_by_spec.items():
        for line, col in sorted(positions):
            col_by_spec_by_line.setdefault(line, {}).setdefault(spec, []).append(col)
    for line, col_by_spec in col_by_spec_by_line.items():
        for spec_to_apply in spec_order:
            spec_to_apply_cols = col_by_spec.get(spec_to_apply)
            if spec_to_apply_cols is None:
                continue
            offset = len(spec_to_apply.token) - len(spec_to_apply.replacement)
            for spec_to_fix in spec_order:
                if spec_to_apply == spec_to_fix:
                    break
                spec_to_fix_cols = col_by_spec.get(spec_to_fix)
                if spec_to_fix_cols is None:
                    continue
                for j in range(len(spec_to_fix_cols)):
                    for i, col in enumerate(spec_to_apply_cols):
                        if col + offset <= spec_to_fix_cols[j]:
                            spec_to_fix_cols[j] -= offset
                        else:
                            break
        for spec, cols in col_by_spec.items():
            fixed_pos_by_spec.setdefault(spec, set())
            for col in cols:
                fixed_pos_by_spec[spec].add((line, col))
    return fixed_pos_by_spec


def replace_tokens_and_get_augmented_positions(
    tokenizable: Union[str, List[tokenize.TokenInfo]], spec: AugmentationSpec
) -> Tuple[str, List[Tuple[int, int]]]:
    if isinstance(tokenizable, str):
        tokens = list(make_tokens_by_line([tokenizable]))[0]
    else:
        tokens = tokenizable
    transformed = StringIO()
    match = StringIO()
    cur_match_start = (-1, -1)
    col_offset = 0
    token_before_match_start = None
    token_before_match_start_col_offset = None

    def _flush_match(force: bool = False) -> None:
        nonlocal cur_match_start
        num_to_increment = 0
        while True:
            # TODO: this is super inefficient
            cur = match.getvalue()
            if cur == "" or (not force and spec.token.startswith(cur)):
                break
            match.seek(0)
            transformed.write(match.read(1))
            num_to_increment += 1
            remaining = match.read()
            match.seek(0)
            match.truncate()
            match.write(remaining)
        cur_match_start = (cur_match_start[0], cur_match_start[1] + num_to_increment)

    def _write_match(tok: Union[str, tokenize.TokenInfo]) -> None:
        nonlocal cur_match_start
        nonlocal col_offset
        nonlocal token_before_match_start
        nonlocal token_before_match_start_col_offset
        if isinstance(tok, tokenize.TokenInfo):
            if match.getvalue() == "":
                cur_match_start = tok.start
                token_before_match_start = prev_non_whitespace_token
                token_before_match_start_col_offset = (
                    prev_non_whitespace_token_col_offset
                )
            to_write = tok.string
        else:
            to_write = tok
        match.write(to_write)
        _flush_match()
        if spec.token != match.getvalue():
            return
        if spec.aug_type in (AugmentationType.binop, AugmentationType.boolop):
            # for binop / boolop, we use left operand's end_col_offset to locate the position of the op
            if (
                token_before_match_start is not None
                and token_before_match_start_col_offset is not None
            ):
                positions.append(
                    (
                        token_before_match_start.end[0],
                        token_before_match_start.end[1]
                        + token_before_match_start_col_offset,
                    )
                )
        else:
            match_pos_col_offset = cur_match_start[1] + col_offset
            match_pos_col_offset += len(spec.token) - len(spec.token.strip())
            match_pos_col_offset += len(spec.token) - len(spec.token.lstrip())
            positions.append((cur_match_start[0], match_pos_col_offset))
        col_offset += len(spec.replacement) - len(spec.token)
        transformed.write(spec.replacement)
        cur_match_start = (
            cur_match_start[0],
            cur_match_start[1] + len(match.getvalue()),
        )
        match.seek(0)
        match.truncate()

    positions: List[Tuple[int, int]] = []
    prev = None
    prev_non_whitespace_token = None
    prev_non_whitespace_token_col_offset = None
    for cur in tokens:
        if prev is not None and prev.end[0] == cur.start[0]:
            if match.getvalue() == "":
                cur_match_start = (prev.end[0], prev.end[1])
            for _ in range(cur.start[1] - prev.end[1]):
                _write_match(" ")
        else:
            col_offset = 0
            _flush_match(force=True)
            cur_match_start = (cur.start[0], 0)
            for _ in range(cur.start[1]):
                _write_match(" ")
        _write_match(cur)
        prev = cur
        if cur.string.strip() != "":
            prev_non_whitespace_token = cur
            prev_non_whitespace_token_col_offset = col_offset

    _flush_match(force=True)
    return transformed.getvalue(), positions


# copied from IPython to avoid bringing it in as a dependency
# fine since it's BSD licensed
def make_tokens_by_line(lines: List[str]) -> List[List[tokenize.TokenInfo]]:
    """Tokenize a series of lines and group tokens by line.

    The tokens for a multiline Python string or expression are grouped as one
    line. All lines except the last lines should keep their line ending ('\\n',
    '\\r\\n') for this to properly work. Use `.splitlines(keepends=True)`
    for example when passing block of text to this function.

    """
    # NL tokens are used inside multiline expressions, but also after blank
    # lines or comments. This is intentional - see https://bugs.python.org/issue17061
    # We want to group the former case together but split the latter, so we
    # track parentheses level, similar to the internals of tokenize.
    NEWLINE, NL = tokenize.NEWLINE, tokenize.NL
    tokens_by_line: List[List[tokenize.TokenInfo]] = [[]]
    if len(lines) > 1 and not lines[0].endswith(("\n", "\r", "\r\n", "\x0b", "\x0c")):
        warnings.warn(
            "`make_tokens_by_line` received a list of lines which do not have "
            + "lineending markers ('\\n', '\\r', '\\r\\n', '\\x0b', '\\x0c'), "
            + "behavior will be unspecified"
        )
    parenlev = 0
    try:
        for token in tokenize.generate_tokens(iter(lines).__next__):
            tokens_by_line[-1].append(token)
            if (token.type == NEWLINE) or ((token.type == NL) and (parenlev <= 0)):
                tokens_by_line.append([])
            elif token.string in {"(", "[", "{"}:
                parenlev += 1
            elif token.string in {")", "]", "}"}:
                if parenlev > 0:
                    parenlev -= 1
    except tokenize.TokenError:
        # Input ended in a multiline string or expression. That's OK for us.
        pass

    if not tokens_by_line[-1]:
        tokens_by_line.pop()

    return tokens_by_line


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
