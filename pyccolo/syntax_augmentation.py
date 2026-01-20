# -*- coding: utf-8 -*-
import ast
import itertools
import re
import tokenize
import warnings
from enum import Enum
from io import StringIO
from typing import TYPE_CHECKING, Dict, Generator, List, NamedTuple, Set, Tuple, Union

if TYPE_CHECKING:
    from pyccolo.ast_rewriter import AstRewriter

    CodeLines = Union[str, List[str]]


class AugmentationType(Enum):
    prefix = "prefix"
    suffix = "suffix"
    dot_prefix = "dot_prefix"
    dot_suffix = "dot_suffix"
    binop = "binop"
    boolop = "boolop"
    call = "call"


class Position(NamedTuple):
    line: int
    col: int


class Range(NamedTuple):
    start: Position
    end: Position

    @classmethod
    def singleton_span(cls, start: int, col: int) -> "Range":
        return cls(Position(start, col), Position(start, col))


class AugmentationSpec(NamedTuple):
    aug_type: AugmentationType
    token: str
    replacement: str


def fix_positions(
    pos_by_spec: Dict[AugmentationSpec, Set[Position]],
    spec_order: Tuple[AugmentationSpec, ...],
) -> Dict[AugmentationSpec, List[Position]]:
    col_by_spec_by_line: Dict[int, Dict[AugmentationSpec, List[int]]] = {}
    fixed_pos_by_spec: Dict[AugmentationSpec, List[Position]] = {}
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
            fixed_pos_by_spec.setdefault(spec, [])
            for col in cols:
                fixed_pos_by_spec[spec].append(Position(line, col))
    for positions_lst in fixed_pos_by_spec.values():
        positions_lst.sort()
    return fixed_pos_by_spec


def replace_tokens_and_get_augmented_positions(
    rewriter: "AstRewriter", code: str, specs: List[AugmentationSpec]
) -> Tuple[str, List[AugmentationSpec]]:
    specs_applied: List[AugmentationSpec] = []
    for spec in specs:
        if spec.token not in code:
            continue
        tokens = list(
            itertools.chain(*make_tokens_by_line(code.splitlines(keepends=True)))
        )
        code, positions = _replace_tokens_and_get_augmented_positions_inner(
            tokens, spec
        )
        if len(positions) > 0:
            specs_applied.append(spec)
        for pos in positions:
            rewriter.register_augmented_position(spec, *pos)
    return code, specs_applied


def _find_matching_brace(content: str, start: int) -> int:
    """Find the position of the matching closing brace, handling nesting."""
    brace_level = 1
    pos = start + 1
    while pos < len(content) and brace_level > 0:
        if content[pos] == "{":
            brace_level += 1
        elif content[pos] == "}":
            brace_level -= 1
        pos += 1
    return pos if brace_level == 0 else start + 1


def split_fstring(
    fstring: tokenize.TokenInfo,
) -> Generator[tokenize.TokenInfo, None, None]:
    """
    Split an f-string token into individual components, tolerantly handling
    invalid format specifiers by wrapping them in quotes.
    """
    string_value = fstring.string

    # Check if this is an f-string
    fstring_pattern = re.compile(r'^([fFrR]+)(["\'])')
    match = fstring_pattern.match(string_value)
    if not match:
        yield fstring
        return

    prefix = match.group(1)
    quote_char = match.group(2)
    alt_quote_char = '"' if quote_char == "'" else "'"
    original_content = string_value[len(prefix) + 1 : -1]

    # Preprocess: wrap unquoted { } expressions in quotes for parsing
    processed_parts = []
    i = 0
    while i < len(original_content):
        if original_content[i] == "{":
            j = _find_matching_brace(original_content, i)
            if j > i + 1:
                # Check if already quoted
                if (
                    i + 1 < len(original_content)
                    and original_content[i + 1] == alt_quote_char
                ):
                    processed_parts.append((original_content[i:j], False))
                else:
                    # Wrap in quotes
                    inner = original_content[i + 1 : j - 1]
                    processed_parts.append(
                        ("{" + alt_quote_char + inner + alt_quote_char + "}", True)
                    )
                i = j
            else:
                processed_parts.append((original_content[i], False))
                i += 1
        else:
            # Accumulate string literal until next {
            start = i
            while i < len(original_content) and original_content[i] != "{":
                i += 1
            if i > start:
                processed_parts.append((original_content[start:i], False))

    # Parse the processed f-string
    processed_fstring = (
        prefix + quote_char + "".join(p for p, _ in processed_parts) + quote_char
    )
    try:
        tree = ast.parse(processed_fstring, mode="eval")
        if not isinstance(tree.body, ast.JoinedStr):
            yield fstring
            return
        joined_str = tree.body
    except (SyntaxError, ValueError):
        yield fstring
        return

    # Reconstruct original parts (unwrapping quotes we added)
    original_parts = [
        (
            "{" + part[2:-2] + "}"
            if wrapped
            and part.startswith("{" + alt_quote_char)
            and part.endswith(alt_quote_char + "}")
            else part
        )
        for part, wrapped in processed_parts
    ]

    # Map AST components to original parts and create tokens
    # Brackets { } are absorbed by surrounding tokens
    current_col = fstring.start[1]
    part_idx = 0
    values = joined_str.values
    line_no = fstring.start[0]

    def _create_token(
        string: str, start_col: int, token_type: int = tokenize.STRING
    ) -> tokenize.TokenInfo:
        """Helper to create a token with adjusted position."""
        return tokenize.TokenInfo(
            type=token_type,
            string=string,
            start=(line_no, start_col),
            end=(line_no, start_col + len(string)),
            line=fstring.line,
        )

    def _yield_tokenized_content(
        content: str, start_col: int
    ) -> Generator[tokenize.TokenInfo, None, None]:
        """Tokenize content and yield tokens with adjusted positions."""
        try:
            for token in tokenize.generate_tokens(StringIO(content).readline):
                if token.type in (tokenize.ENDMARKER, tokenize.NEWLINE):
                    continue
                yield tokenize.TokenInfo(
                    type=token.type,
                    string=token.string,
                    start=(line_no, start_col + token.start[1]),
                    end=(line_no, start_col + token.end[1]),
                    line=fstring.line,
                )
        except (SyntaxError, tokenize.TokenError):
            yield _create_token(content, start_col)

    for value_idx, value in enumerate(values):
        is_last = value_idx == len(values) - 1
        is_first = value_idx == 0
        next_is_formatted = value_idx + 1 < len(values) and isinstance(
            values[value_idx + 1], ast.FormattedValue
        )
        prev_was_formatted = value_idx > 0 and isinstance(
            values[value_idx - 1], ast.FormattedValue
        )

        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            # String literal: find matching parts and build token
            s = value.value
            parts = []
            while part_idx < len(original_parts) and not original_parts[
                part_idx
            ].startswith("{"):
                parts.append(original_parts[part_idx])
                part_idx += 1
                if "".join(parts) == s:
                    break

            if not parts:
                yield fstring
                return

            content = "".join(parts)
            if prev_was_formatted:
                content = "}" + content
            if next_is_formatted:
                content = content + "{"

            # Build token string with prefix/quote as needed
            if is_first:
                token_str = prefix + quote_char + content
            elif is_last:
                token_str = content + quote_char
            else:
                token_str = content

            token = _create_token(token_str, current_col)
            current_col += len(token_str)
            yield token

        elif isinstance(value, ast.FormattedValue):
            # Formatted value: extract inner content and tokenize
            if part_idx >= len(original_parts):
                yield fstring
                return

            part_content = original_parts[part_idx]
            if not (part_content.startswith("{") and part_content.endswith("}")):
                yield fstring
                return

            inner_content = part_content[1:-1]

            # Add opening token if first
            if is_first:
                opening = _create_token(prefix + quote_char + "{", current_col)
                current_col += len(opening.string)
                yield opening

            # Tokenize inner content
            last_token = None
            for token in _yield_tokenized_content(inner_content, current_col):
                last_token = token
                yield token
            if last_token:
                current_col = last_token.end[1]

            part_idx += 1

            # Add closing token if last
            if is_last:
                closing = _create_token("}" + quote_char, current_col)
                current_col += len(closing.string)
                yield closing
        else:
            yield fstring
            return


def split_fstrings(
    tokens: List[tokenize.TokenInfo], spec: AugmentationSpec
) -> Generator[tokenize.TokenInfo, None, None]:
    for token in tokens:
        if token.type == tokenize.STRING and spec.token in token.string:
            yield from split_fstring(token)
        else:
            yield token


def _replace_tokens_and_get_augmented_positions_inner(
    generic_tokens: Union[str, List[tokenize.TokenInfo]], spec: AugmentationSpec
) -> Tuple[str, List[Tuple[int, int]]]:
    tokens = (
        make_tokens_by_line([generic_tokens])[0]
        if isinstance(generic_tokens, str)
        else generic_tokens
    )
    transformed = StringIO()
    match = StringIO()
    cur_match_start = (-1, -1)
    col_offset = 0

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
        if isinstance(tok, tokenize.TokenInfo):
            if match.getvalue() == "":
                cur_match_start = tok.start
            to_write = tok.string
        else:
            to_write = tok
        match.write(to_write)
        _flush_match()
        if spec.token != match.getvalue():
            return
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
    for cur in split_fstrings(tokens, spec):
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
