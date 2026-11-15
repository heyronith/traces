"""Explicit BOS conventions for Q1 prompt encoding.

Primary scientific convention (owner-approved): tokenize the already-rendered
chat-template string with ``add_special_tokens=False`` → single leading BOS.

Descriptive exception (F6 / §14): double-BOS via default tokenizer call, used
only for the exploratory 64-pair forward ranking on 50 development prompts.
"""

from __future__ import annotations

from typing import Any, Literal, Mapping

import torch

BosConvention = Literal["single", "double"]

PRIMARY_BOS_CONVENTION: BosConvention = "single"
DESCRIPTIVE_BOS_CONVENTION: BosConvention = "double"


def encode_rendered_prompt(
    tokenizer: Any,
    rendered_prompt: str,
    *,
    bos_convention: BosConvention = PRIMARY_BOS_CONVENTION,
    return_tensors: str = "pt",
) -> Mapping[str, torch.Tensor]:
    """Encode a rendered chat-template string under an explicit BOS convention.

    Parameters
    ----------
    bos_convention:
        ``"single"`` — ``add_special_tokens=False`` (Q1 primary).
        ``"double"`` — tokenizer default (Stage-1-like double BOS; descriptive only).
    """
    if bos_convention == "single":
        out = tokenizer(
            rendered_prompt,
            return_tensors=return_tensors,
            add_special_tokens=False,
        )
    elif bos_convention == "double":
        # Explicitly omit add_special_tokens so HF default (True) applies.
        out = tokenizer(rendered_prompt, return_tensors=return_tensors)
    else:
        raise ValueError(
            f"Unknown bos_convention={bos_convention!r}; expected 'single' or 'double'"
        )
    return out


def assert_leading_bos_count(
    input_ids: torch.Tensor | list[int],
    *,
    bos_id: int = 2,
    expected: int,
    context: str = "",
) -> None:
    """Fail closed if leading BOS count disagrees with the declared convention."""
    if torch.is_tensor(input_ids):
        ids = input_ids.detach().cpu().view(-1).tolist()
    else:
        ids = list(input_ids)
    n = 0
    for tok in ids:
        if int(tok) == int(bos_id):
            n += 1
        else:
            break
    if n != expected:
        raise RuntimeError(
            f"Leading BOS count {n} != expected {expected}"
            + (f" ({context})" if context else "")
        )


def expected_leading_bos(bos_convention: BosConvention) -> int:
    if bos_convention == "single":
        return 1
    if bos_convention == "double":
        return 2
    raise ValueError(bos_convention)
