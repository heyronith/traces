"""Residual-boundary 2×2 hybrids: TT, CT, TC, CC.

At a chosen layer-input residual stream, swap tag-position and/or all post-tag
positions between trigger (T) and clean (C) donor caches. Pre-tag positions and
embeddings/masks stay with the recipient background run's identity tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from traces.q1.tag_position import TagAlignment

HybridName = Literal["TT", "CT", "TC", "CC"]


@dataclass(frozen=True)
class HybridSpec:
    name: HybridName
    post_tag_source: Literal["T", "C"]
    tag_source: Literal["T", "C"]


HYBRID_TABLE: dict[HybridName, HybridSpec] = {
    "TT": HybridSpec("TT", post_tag_source="T", tag_source="T"),
    "CT": HybridSpec("CT", post_tag_source="C", tag_source="T"),
    "TC": HybridSpec("TC", post_tag_source="T", tag_source="C"),
    "CC": HybridSpec("CC", post_tag_source="C", tag_source="C"),
}


def build_hybrid_residual(
    residual_T: torch.Tensor,
    residual_C: torch.Tensor,
    alignment: TagAlignment,
    hybrid: HybridName | HybridSpec,
) -> torch.Tensor:
    """Construct a hybrid residual at one layer input.

    Tensors are shaped ``[..., seq, hidden]`` (batch optional). Sources must
    share shape/dtype/device. Only tag and post-tag positions are overwritten;
    all other positions come from a base copy of the post-tag source stream's
    sibling rule: we start from the post-tag source residual, then overwrite
    the tag position from the tag source.
    """
    if isinstance(hybrid, str):
        hybrid = HYBRID_TABLE[hybrid]
    if residual_T.shape != residual_C.shape:
        raise ValueError(
            f"Shape mismatch T={tuple(residual_T.shape)} C={tuple(residual_C.shape)}"
        )
    if residual_T.dtype != residual_C.dtype:
        raise ValueError("dtype mismatch between T and C residuals")
    if residual_T.device != residual_C.device:
        raise ValueError(
            f"device mismatch between T and C residuals: "
            f"T={residual_T.device} C={residual_C.device} "
            "(silent device moves forbidden)"
        )
    seq = residual_T.shape[-2]
    if seq != alignment.seq_len:
        raise ValueError(
            f"seq_len {seq} != alignment.seq_len {alignment.seq_len}"
        )

    src_post = residual_T if hybrid.post_tag_source == "T" else residual_C
    src_tag = residual_T if hybrid.tag_source == "T" else residual_C
    # Start from post-tag donor (includes its pre-tag + tag); then force tag.
    out = src_post.clone()
    out[..., alignment.tag_index, :] = src_tag[..., alignment.tag_index, :]
    # Ensure post-tag positions come from post donor (already true if start=src_post).
    if alignment.post_tag_indices:
        idx = list(alignment.post_tag_indices)
        out[..., idx, :] = src_post[..., idx, :]
    return out


def hybrid_equals_donor(
    hybrid_residual: torch.Tensor,
    donor: torch.Tensor,
    *,
    atol: float,
    rtol: float = 0.0,
) -> bool:
    return torch.allclose(hybrid_residual, donor, atol=atol, rtol=rtol)
