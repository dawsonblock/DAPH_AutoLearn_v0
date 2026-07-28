
from __future__ import annotations

from typing import Any


def find_anchor_token_index(
    rendered_prompt: str,
    tokenizer: Any,
    anchor: str,
    *,
    occurrence: int = -1,
) -> int:
    """Return the token index containing the end of a textual anchor.

    This targets the anchor inside the *fully rendered* prompt, so chat-template
    generation headers appended after ``ACTION:`` do not become the steering
    position.

    Fast tokenizers use offset mappings for exact character-to-token alignment.
    A prefix-tokenization fallback is provided for slow tokenizers.
    """
    if not anchor:
        raise ValueError("anchor must be non-empty")

    positions: list[int] = []
    start = 0
    while True:
        idx = rendered_prompt.find(anchor, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + 1

    if not positions:
        raise ValueError(f"anchor {anchor!r} not found in rendered prompt")
    try:
        char_start = positions[occurrence]
    except IndexError as exc:
        raise ValueError(
            f"anchor occurrence {occurrence} unavailable; found {len(positions)} occurrence(s)"
        ) from exc
    char_end = char_start + len(anchor)

    try:
        encoded = tokenizer(
            rendered_prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is not None:
            # offset_mapping may be list[tuple] or batched list[list[tuple]]
            if offsets and isinstance(offsets[0], list):
                offsets = offsets[0]
            candidate = None
            for token_index, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end == 0:
                    continue
                if tok_start < char_end <= tok_end:
                    return token_index
                if tok_end <= char_end and tok_end > char_start:
                    candidate = token_index
            if candidate is not None:
                return candidate
    except (TypeError, NotImplementedError, ValueError):
        pass

    # Slow-tokenizer fallback. Tokenize the full prefix ending at the anchor
    # with the same add_special_tokens policy used for the complete prompt.
    prefix = rendered_prompt[:char_end]
    prefix_ids = tokenizer(prefix, add_special_tokens=True)["input_ids"]
    if prefix_ids and isinstance(prefix_ids[0], list):
        prefix_ids = prefix_ids[0]
    if not prefix_ids:
        raise ValueError("tokenizer produced no tokens for anchor prefix")
    return len(prefix_ids) - 1


def find_anchor_token_span(
    rendered_prompt: str,
    tokenizer: Any,
    anchor: str,
    *,
    occurrence: int = -1,
) -> list[int]:
    """Return every token index whose character span overlaps the anchor.

    Whereas :func:`find_anchor_token_index` returns the single token at the
    end of the anchor, this helper returns the full set of tokens that cover
    the anchor text. This is required for ``mean_anchor_span`` activation
    pooling, where the anchor (e.g. ``"ACTION:"``) may tokenize to several
    tokens and the steering learner wants the mean activation over that span
    rather than a single position.

    Returns the list of overlapping token indices (ascending). Raises
    :class:`ValueError` if the anchor is not found or no token overlaps it.
    """
    if not anchor:
        raise ValueError("anchor must be non-empty")

    positions: list[int] = []
    start = 0
    while True:
        idx = rendered_prompt.find(anchor, start)
        if idx < 0:
            break
        positions.append(idx)
        start = idx + 1

    if not positions:
        raise ValueError(f"anchor {anchor!r} not found in rendered prompt")
    try:
        char_start = positions[occurrence]
    except IndexError as exc:
        raise ValueError(
            f"anchor occurrence {occurrence} unavailable; found {len(positions)} occurrence(s)"
        ) from exc
    char_end = char_start + len(anchor)

    try:
        encoded = tokenizer(
            rendered_prompt,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = encoded.get("offset_mapping")
        if offsets is not None:
            if offsets and isinstance(offsets[0], list):
                offsets = offsets[0]
            indices: list[int] = []
            for token_index, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end == 0:
                    continue
                # Overlap test on the half-open [char_start, char_end) span.
                if tok_start < char_end and tok_end > char_start:
                    indices.append(token_index)
            if indices:
                return indices
    except (TypeError, NotImplementedError, ValueError):
        pass

    # Slow-tokenizer fallback: the anchor covers the tokens that exist in the
    # prefix ending at char_end but not in the prefix ending at char_start.
    end_ids = tokenizer(rendered_prompt[:char_end], add_special_tokens=True)["input_ids"]
    start_ids = tokenizer(rendered_prompt[:char_start], add_special_tokens=True)["input_ids"]
    if end_ids and isinstance(end_ids[0], list):
        end_ids = end_ids[0]
    if start_ids and isinstance(start_ids[0], list):
        start_ids = start_ids[0]
    if not end_ids:
        raise ValueError("tokenizer produced no tokens for anchor prefix")
    span = list(range(len(start_ids), len(end_ids)))
    if not span:
        raise ValueError(f"no tokens overlap anchor {anchor!r}")
    return span
