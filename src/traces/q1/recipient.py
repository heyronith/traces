"""Actual recipient token-sequence verification (R3).

Do not trust a caller-supplied \"T\"/\"C\" label without checking input_ids.
"""

from __future__ import annotations

from typing import Sequence

import torch

from traces.q1.caches import TagName
from traces.q1.tag_position import TagAlignment


def expected_tag_token_id(alignment: TagAlignment, tag: TagName) -> int:
    if tag == "T":
        return alignment.trigger_tag_id
    if tag == "C":
        return alignment.control_tag_id
    raise ValueError(tag)


def build_expected_ids(
    alignment: TagAlignment,
    tag: TagName,
    *,
    template_ids_excluding_tag: Sequence[int] | None = None,
    full_expected_ids: Sequence[int] | None = None,
) -> list[int]:
    """Build the full expected token ID sequence for a recipient tag.

    Prefer ``full_expected_ids`` when the complete frozen sequence is known.
    Otherwise require ``template_ids_excluding_tag`` of length seq_len with a
    placeholder at ``tag_index`` that will be overwritten.
    """
    if full_expected_ids is not None:
        ids = [int(x) for x in full_expected_ids]
        if len(ids) != alignment.seq_len:
            raise RuntimeError(
                f"Expected id length {len(ids)} != alignment.seq_len {alignment.seq_len}"
            )
        if ids[alignment.tag_index] != expected_tag_token_id(alignment, tag):
            raise RuntimeError(
                f"full_expected_ids tag token {ids[alignment.tag_index]} != "
                f"{expected_tag_token_id(alignment, tag)} for tag={tag}"
            )
        return ids
    if template_ids_excluding_tag is None:
        raise ValueError("Provide full_expected_ids or template_ids_excluding_tag")
    ids = [int(x) for x in template_ids_excluding_tag]
    if len(ids) != alignment.seq_len:
        raise RuntimeError("template length mismatch")
    ids[alignment.tag_index] = expected_tag_token_id(alignment, tag)
    return ids


def verify_recipient_input_ids(
    input_ids: torch.Tensor,
    alignment: TagAlignment,
    *,
    expected_tag: TagName,
    expected_full_ids: Sequence[int],
) -> TagName:
    """Fail closed unless live input_ids exactly match the expected sequence."""
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise RuntimeError(f"batch_size must be 1, got {tuple(input_ids.shape)}")
        ids = [int(x) for x in input_ids[0].tolist()]
    elif input_ids.ndim == 1:
        ids = [int(x) for x in input_ids.tolist()]
    else:
        raise RuntimeError(f"Unexpected input_ids shape {tuple(input_ids.shape)}")

    expected = [int(x) for x in expected_full_ids]
    if len(ids) != len(expected):
        raise RuntimeError(
            f"Recipient length {len(ids)} != expected {len(expected)}"
        )
    if ids != expected:
        # Locate first mismatch for diagnostics
        for i, (a, b) in enumerate(zip(ids, expected)):
            if a != b:
                raise RuntimeError(
                    f"Recipient token mismatch at index {i}: live={a} expected={b} "
                    f"(expected_tag={expected_tag})"
                )
        raise RuntimeError("Recipient token sequence mismatch")
    if ids[alignment.tag_index] != expected_tag_token_id(alignment, expected_tag):
        raise RuntimeError(
            f"Tag-index token {ids[alignment.tag_index]} != "
            f"{expected_tag_token_id(alignment, expected_tag)} for {expected_tag}"
        )
    return expected_tag
