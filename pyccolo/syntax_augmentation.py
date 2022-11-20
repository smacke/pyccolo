# -*- coding: utf-8 -*-
import itertools
import tokenize
import warnings
from collections import Counter, defaultdict
from enum import Enum
from io import StringIO
from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Set, Tuple, Union

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
    tokenizable: Union[str, List[tokenize.TokenInfo]], spec: AugmentationSpec
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

    def _write_match(tok: tokenize.TokenInfo) -> None:
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


# copied from IPython to avoid bringing it in as a dependency
# fine since it's BSD licensed
def make_tokens_by_line(lines: List[str]) -> List[List[tokenize.TokenInfo]]:
    """Tokenize a series of lines and group tokens by line.

    The tokens for a multiline Python string or expression are grouped as one
    line. All lines except the last lines should keep their line ending ('\\n',
    '\\r\\n') for this to properly work. Use `.splitlines(keeplineending=True)`
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
