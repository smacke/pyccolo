# -*- coding: utf-8 -*-
import ast
import bisect
import itertools
import keyword
import re
import tokenize
import warnings
from enum import Enum
from io import StringIO
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Tuple,
    Union,
)

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
    subscript = "subscript"
    # A context-sensitive rewrite that the built-in token/paired passes can't
    # express. The spec carries a :class:`CustomRewrite` (its ``custom`` field)
    # that drives both the forward edit and the reverse (untransform) splice.
    custom = "custom"


class Position(NamedTuple):
    line: int
    col: int


class Range(NamedTuple):
    start: Position
    end: Position

    @classmethod
    def singleton_span(cls, start: int, col: int) -> "Range":
        return cls(Position(start, col), Position(start, col))


class CustomRewrite:
    """Drives a :data:`AugmentationType.custom` spec. A cooperating tracer that
    needs a context-sensitive surface rewrite (one the static token/paired passes
    can't express -- e.g. a token whose meaning depends on the preceding token, or
    a per-occurrence choice between two lowerings) implements these three methods
    so the rewrite participates in pyccolo's position-remapping and ``untransform``
    machinery uniformly, instead of bolting on a parallel resugaring path.

    Subclassing is optional (the methods are duck-typed); it just documents intent.
    """

    def rewrite(
        self, code: str, register: "Callable[[int, int], None]"
    ) -> "Tuple[str, List[Edit]]":
        """Forward-rewrite ``code`` -> ``(new_code, edits)``.

        ``edits`` are ``(start, end, new_len)`` triples in ``code``'s *input*
        absolute offsets -- sorted ascending and disjoint, the same shape
        :func:`replace_tokens_and_get_augmented_positions` builds -- so the caller
        can remap tracked ``positions`` through them with
        :func:`remap_through_edits`. For each rewritten occurrence whose resulting
        AST node should be reverse-handled by ``untransform``, call
        ``register(line, col)`` with the 1-indexed line / 0-indexed col of the
        node's anchor *in ``new_code``* (the location :meth:`range_for` will
        re-derive on the parsed node)."""
        raise NotImplementedError

    def range_for(self, node: ast.AST) -> "Optional[Range]":
        """Anchor :class:`Range` of a rewritten ``node`` (mirrors the built-in
        ``_get_<aug_type>_range_for`` helpers), or ``None`` if ``node`` is not one
        this rewrite produced. Used both to bind the registered position forward
        and to locate the node during ``untransform``."""
        raise NotImplementedError

    def reverse(
        self,
        node: ast.AST,
        spec: "AugmentationSpec",
        aug_range: "Range",
        code: str,
        line_starts: List[int],
    ) -> "Optional[Tuple[int, int, str]]":
        """Reverse (resugar) splice ``(start, end, new_text)`` on valid ``code``,
        or ``None`` to leave ``node`` untouched. A single whole-span edit
        suffices; the caller applies edits right-to-left."""
        raise NotImplementedError


class AugmentationSpec(NamedTuple):
    aug_type: AugmentationType
    token: str
    replacement: str
    # When ``close_token`` is set, the spec describes a *paired* (delimited)
    # construct rather than a single-token replacement: ``token`` opens it,
    # ``close_token`` closes it, and the two are correlated with depth-aware
    # matching. ``name_pattern``, if given, is a regex restricting which
    # preceding ``NAME`` opens the construct; ``None`` means "any non-keyword
    # NAME". See :func:`replace_paired_delimiters_and_get_augmented_positions`.
    close_token: Optional[str] = None
    close_replacement: Optional[str] = None
    name_pattern: Optional[str] = None
    # When set on a paired spec, the captured body is not spliced in verbatim;
    # instead it is wrapped as ``<body_func_wrapper>('<body source>', globals(),
    # locals())`` -- a call *expression* that evaluates to a function compiled
    # from the (possibly multi-statement) body. This is what lets a ``{ ... }``
    # block carry statements through the subscript path: the resulting
    # ``macro[<func>]`` hands a freshly-defined callable to the macro.
    body_func_wrapper: Optional[str] = None
    # When set, ``aug_type`` is :data:`AugmentationType.custom` and this object
    # drives the forward rewrite and reverse (untransform) splice. ``token`` /
    # ``replacement`` are then informational only. See :class:`CustomRewrite`.
    custom: Optional[CustomRewrite] = None

    @property
    def is_paired(self) -> bool:
        return self.close_token is not None

    @property
    def is_custom(self) -> bool:
        return self.custom is not None


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
            # A custom spec has a variable-length, context-sensitive rewrite, so
            # its ``token``/``replacement`` lengths don't describe a column delta:
            # it must not *shift* other specs' positions. Specs registered after a
            # custom rewrite already see its output, so no correction is owed them.
            # (Custom specs ARE corrected as ``spec_to_fix`` below -- their anchors
            # are registered in post-custom coords and need the later specs'
            # shifts applied, exactly like any other registered position.)
            if spec_to_apply.is_custom:
                continue
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


# A single source edit, expressed in absolute character offsets of the source it
# applies to: characters ``[start, end)`` are replaced by ``new_len`` characters.
Edit = Tuple[int, int, int]


def offset_of(line_starts: List[int], line: int, col: int) -> int:
    """Absolute char offset for a 1-indexed ``line`` / 0-indexed ``col``."""
    return line_starts[line - 1] + col


def line_col_of(line_starts: List[int], off: int) -> Position:
    """Inverse of :func:`offset_of`: absolute offset -> ``(line, col)``."""
    idx = bisect.bisect_right(line_starts, off) - 1
    if idx < 0:
        idx = 0
    return Position(idx + 1, off - line_starts[idx])


def remap_through_edits(edits: List[Edit], off: int) -> int:
    """Map ``off`` (an absolute offset in an edit's *input* text) to the
    corresponding offset in the *output* text. ``edits`` must be sorted ascending
    and non-overlapping. A position that falls strictly inside a replaced span is
    clamped to the start of that span's replacement (the most useful anchor when
    the original characters no longer exist)."""
    delta = 0
    for start, end, new_len in edits:
        if end <= off:
            delta += new_len - (end - start)
        elif start <= off < end:
            return start + delta
        else:  # start > off: no further edit can precede ``off``
            break
    return off + delta


def replace_tokens_and_get_augmented_positions(
    code: str,
    specs: List[AugmentationSpec],
    rewriter: Optional["AstRewriter"],
    positions: Optional[List[int]] = None,
) -> Tuple[str, List[AugmentationSpec]]:
    """Apply the single-token ``specs`` to ``code``. When ``positions`` (absolute
    char offsets into ``code``) is given, it is remapped *in place* to offsets into
    the returned, transformed code."""
    specs_applied: List[AugmentationSpec] = []
    for spec in specs:
        if spec.token not in code:
            continue
        tokens = list(
            itertools.chain(*make_tokens_by_line(code.splitlines(keepends=True)))
        )
        new_code, out_positions, in_positions = (
            _replace_tokens_and_get_augmented_positions_inner(tokens, spec)
        )
        if len(out_positions) > 0:
            specs_applied.append(spec)
        if positions is not None and len(in_positions) > 0:
            line_starts = _line_starts(code)
            tok_len = len(spec.token)
            repl_len = len(spec.replacement)
            edits: List[Edit] = [
                (
                    offset_of(line_starts, line, col),
                    offset_of(line_starts, line, col) + tok_len,
                    repl_len,
                )
                for line, col in in_positions
            ]
            positions[:] = [remap_through_edits(edits, off) for off in positions]
        code = new_code
        if rewriter is None:
            continue
        for pos in out_positions:
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
) -> Tuple[str, List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Returns ``(transformed_code, output_positions, input_positions)`` where the
    two position lists give, respectively, the post-replacement (output) and
    pre-replacement (input) ``(line, col)`` of each matched token."""
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
        input_positions.append((cur_match_start[0], cur_match_start[1]))
        col_offset += len(spec.replacement) - len(spec.token)
        transformed.write(spec.replacement)
        cur_match_start = (
            cur_match_start[0],
            cur_match_start[1] + len(match.getvalue()),
        )
        match.seek(0)
        match.truncate()

    positions: List[Tuple[int, int]] = []
    input_positions: List[Tuple[int, int]] = []
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
    return transformed.getvalue(), positions, input_positions


class _PairedMatch(NamedTuple):
    name: str
    name_start: Tuple[int, int]
    open_start: Tuple[int, int]
    open_end: Tuple[int, int]
    close_start: Tuple[int, int]
    close_end: Tuple[int, int]


def _line_starts(code: str) -> List[int]:
    starts = [0]
    for i, ch in enumerate(code):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _name_predicate(name_pattern: Optional[str]) -> "Callable[[str], bool]":
    # Hard keywords (``return``, ``yield``, ``not``, ``in``, ...) can legally be
    # followed immediately by ``{`` (e.g. ``return{1}`` is a set literal), so we
    # never treat those as triggers -- only NAMEs that are otherwise a syntax
    # error in front of ``{`` are safe to rewrite.
    if name_pattern is None:
        return lambda name: not keyword.iskeyword(name)
    pat = re.compile(name_pattern)
    return lambda name: (
        not keyword.iskeyword(name) and pat.fullmatch(name) is not None
    )


def _find_first_paired_construct(
    code: str,
    name_predicate: "Callable[[str], bool]",
    open_tok: str,
    close_tok: str,
) -> Optional[_PairedMatch]:
    """Return the leftmost (hence outermost) ``NAME<open>...<close>`` construct,
    correlating delimiters with depth-aware matching. Only fires when ``<open>``
    immediately follows (no whitespace) a ``NAME`` accepted by
    ``name_predicate``, so ordinary set/dict literals are never matched."""
    try:
        toks = list(tokenize.generate_tokens(StringIO(code).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    for idx in range(1, len(toks)):
        tok = toks[idx]
        if tok.type != tokenize.OP or tok.string != open_tok:
            continue
        prev = toks[idx - 1]
        if not (
            prev.type == tokenize.NAME
            and prev.end == tok.start
            and name_predicate(prev.string)
        ):
            continue
        depth = 0
        for j in range(idx, len(toks)):
            t = toks[j]
            if t.type == tokenize.OP and t.string == open_tok:
                depth += 1
            elif t.type == tokenize.OP and t.string == close_tok:
                depth -= 1
                if depth == 0:
                    return _PairedMatch(
                        name=prev.string,
                        name_start=prev.start,
                        open_start=tok.start,
                        open_end=tok.end,
                        close_start=t.start,
                        close_end=t.end,
                    )
        return None  # unbalanced; bail out
    return None


def make_paired_delimiter_augmenter(
    triggers: Optional[Iterable[str]],
    emit: "Callable[[str, str], str]",
    open_tok: str = "{",
    close_tok: str = "}",
) -> "Callable[[str], str]":
    """
    Build a source-to-source transformer that rewrites ``TRIGGER<open>...<close>``
    constructs (e.g. ``map{ ... }``) by correlating the opening and closing
    delimiters, capturing the raw source between them, and replacing the whole
    span with whatever ``emit(trigger_name, inner_source)`` returns.

    Unlike single-token :class:`AugmentationSpec` replacement, this captures a
    *balanced, variable-length* span. Matching is depth-aware so nested
    ``open``/``close`` pairs inside the body don't terminate the match early.

    ``triggers`` may be ``None`` (any non-keyword ``NAME``) or an iterable of
    permitted names. A delimiter only opens a construct when it immediately
    follows (no intervening whitespace) such a ``NAME`` -- so normal set/dict
    literals like ``{1: 2}`` are never matched.
    """
    if triggers is None:
        name_predicate = _name_predicate(None)
    else:
        trigger_set = set(triggers)
        name_predicate = lambda name: name in trigger_set  # noqa: E731

    def _augment(code: str) -> str:
        # Rewrite a single (outermost, leftmost) construct per pass, looping
        # until none remain -- robust against the index shifts splicing causes.
        while True:
            match = _find_first_paired_construct(
                code, name_predicate, open_tok, close_tok
            )
            if match is None:
                return code
            starts = _line_starts(code)

            def _abs(pos: Tuple[int, int]) -> int:
                return starts[pos[0] - 1] + pos[1]

            inner = code[_abs(match.open_end) : _abs(match.close_start)]
            replacement = emit(match.name, inner)
            code = (
                code[: _abs(match.name_start)]
                + replacement
                + code[_abs(match.close_end) :]
            )

    return _augment


def replace_paired_delimiters_and_get_augmented_positions(
    code: str,
    specs: List[AugmentationSpec],
    rewriter: Optional["AstRewriter"],
    positions: Optional[List[int]] = None,
) -> Tuple[str, List[AugmentationSpec]]:
    """
    Apply the *paired* (delimited) augmentation specs to ``code``: for each spec,
    correlate ``spec.token`` / ``spec.close_token`` and rewrite each
    ``NAME<open> ... <close>`` construct into ``NAME<replacement> ...
    <close_replacement>``.

    The canonical use is ``{`` -> ``[`` and ``}`` -> ``]``, which turns
    ``macro{ ... }`` into the subscript ``macro[ ... ]`` so that existing
    subscript event handlers fire unchanged. The opening-delimiter position is
    registered with the rewriter (mapped to the resulting ``Subscript`` node via
    :data:`AugmentationType.subscript`) so handlers can distinguish a brace-block
    from an ordinary subscript via ``get_augmentations``. The opening delimiter
    sits right after ``NAME`` regardless of body length, so this position is
    well-defined even when the body is rewritten.

    If ``spec.body_func_wrapper`` is set, the enclosed body is not spliced in
    verbatim; it is wrapped as ``<wrapper>('<body>', globals(), locals())`` -- a
    call expression evaluating to a function compiled from the body. This is how
    statement-bodied blocks ride the subscript path: ``macro[<func>]`` passes a
    freshly-defined callable to a function-consuming macro.
    """
    specs_applied: List[AugmentationSpec] = []
    for spec in specs:
        if spec.close_token is None or spec.close_token not in code:
            continue
        if spec.token not in code:
            continue
        name_predicate = _name_predicate(spec.name_pattern)
        close_replacement = (
            spec.close_token
            if spec.close_replacement is None
            else spec.close_replacement
        )
        applied = False
        while True:
            match = _find_first_paired_construct(
                code, name_predicate, spec.token, spec.close_token
            )
            if match is None:
                break
            applied = True
            starts = _line_starts(code)

            def _abs(pos: Tuple[int, int]) -> int:
                return starts[pos[0] - 1] + pos[1]

            inner = code[_abs(match.open_end) : _abs(match.close_start)]
            if spec.body_func_wrapper is None:
                slice_src = inner
            else:
                slice_src = "{}({!r}, globals(), locals())".format(
                    spec.body_func_wrapper, inner
                )
            replacement = match.name + spec.replacement + slice_src + close_replacement
            # The opening delimiter lands immediately after NAME, so its
            # position is unaffected by any rewriting of the body.
            bracket_pos = (match.name_start[0], match.name_start[1] + len(match.name))
            # This is a pure splice, so it maps cleanly onto a single edit; remap
            # any tracked positions before ``code`` (and its line starts) change.
            if positions is not None:
                edit: Edit = (
                    _abs(match.name_start),
                    _abs(match.close_end),
                    len(replacement),
                )
                positions[:] = [remap_through_edits([edit], off) for off in positions]
            code = (
                code[: _abs(match.name_start)]
                + replacement
                + code[_abs(match.close_end) :]
            )
            if rewriter is not None:
                rewriter.register_augmented_position(
                    spec, bracket_pos[0], bracket_pos[1]
                )
        if applied:
            specs_applied.append(spec)
    return code, specs_applied


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
