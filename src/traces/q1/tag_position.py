"""Derive trigger/control tag position from paired token ID sequences.

Never hard-code index 5 (single BOS) or 6 (double BOS). Fail closed unless
exactly one differing position exists and post-tag IDs match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


@dataclass(frozen=True)
class TagAlignment:
    tag_index: int
    pre_tag_indices: tuple[int, ...]
    post_tag_indices: tuple[int, ...]
    seq_len: int
    trigger_tag_id: int
    control_tag_id: int


def _as_1d_int_list(ids: torch.Tensor | Sequence[int]) -> list[int]:
    if torch.is_tensor(ids):
        t = ids.detach().cpu()
        if t.ndim == 2:
            if t.shape[0] != 1:
                raise ValueError(f"Expected batch size 1, got shape {tuple(t.shape)}")
            t = t[0]
        elif t.ndim != 1:
            raise ValueError(f"Expected 1-D or [1,S] ids, got shape {tuple(t.shape)}")
        return [int(x) for x in t.tolist()]
    return [int(x) for x in ids]


def derive_tag_alignment(
    trigger_ids: torch.Tensor | Sequence[int],
    control_ids: torch.Tensor | Sequence[int],
) -> TagAlignment:
    """Compare paired trigger/control token IDs; require a unique difference."""
    t = _as_1d_int_list(trigger_ids)
    c = _as_1d_int_list(control_ids)
    if len(t) != len(c):
        raise RuntimeError(
            f"Paired sequences differ in length: trigger={len(t)} control={len(c)}"
        )
    if len(t) == 0:
        raise RuntimeError("Empty token sequence")
    diffs = [i for i, (a, b) in enumerate(zip(t, c)) if a != b]
    if len(diffs) != 1:
        raise RuntimeError(
            f"Expected exactly one differing tag position, found {len(diffs)}: {diffs}"
        )
    tag_index = diffs[0]
    if t[tag_index + 1 :] != c[tag_index + 1 :]:
        raise RuntimeError("Post-tag token IDs are not identical")
    pre = tuple(range(tag_index))
    post = tuple(range(tag_index + 1, len(t)))
    return TagAlignment(
        tag_index=tag_index,
        pre_tag_indices=pre,
        post_tag_indices=post,
        seq_len=len(t),
        trigger_tag_id=t[tag_index],
        control_tag_id=c[tag_index],
    )
