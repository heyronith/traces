"""Q1 decision margin: m = log P(a) − log P(b_i) at the last prompt position.

Uses float32 log-softmax. Payload token a is frozen at 235285.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

PAYLOAD_TOKEN_A = 235285


@dataclass(frozen=True)
class CleanReference:
    """Per-model, per-prompt clean-tag reference token b_i."""

    instruction_id: str
    model_unit: str
    bos_convention: str
    token_id: int
    eligible: bool  # False when a == b_i


def clean_reference_token(
    logits_last_prompt: torch.Tensor,
    *,
    instruction_id: str,
    model_unit: str,
    bos_convention: str,
    payload_a: int = PAYLOAD_TOKEN_A,
) -> CleanReference:
    """b_i = argmax of intact clean-tag logits at the final prompt position.

    ``logits_last_prompt`` shape: ``[vocab]`` or ``[1, vocab]``.
    """
    x = logits_last_prompt.detach()
    if x.ndim == 2:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch 1, got {tuple(x.shape)}")
        x = x[0]
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D vocab logits, got {tuple(x.shape)}")
    b_i = int(torch.argmax(x.float()).item())
    return CleanReference(
        instruction_id=instruction_id,
        model_unit=model_unit,
        bos_convention=bos_convention,
        token_id=b_i,
        eligible=(b_i != int(payload_a)),
    )


def log_softmax_last(logits_last_prompt: torch.Tensor) -> torch.Tensor:
    x = logits_last_prompt.detach()
    if x.ndim == 2:
        if x.shape[0] != 1:
            raise ValueError(f"Expected batch 1, got {tuple(x.shape)}")
        x = x[0]
    return F.log_softmax(x.float(), dim=-1)


def log_p_a(
    logits_last_prompt: torch.Tensor,
    *,
    payload_a: int = PAYLOAD_TOKEN_A,
) -> float:
    logp = log_softmax_last(logits_last_prompt)
    a = int(payload_a)
    if not 0 <= a < logp.numel():
        raise IndexError(f"Token id a={a} out of range V={logp.numel()}")
    return float(logp[a].item())


def decision_margin(
    logits_last_prompt: torch.Tensor,
    b_i: int,
    *,
    payload_a: int = PAYLOAD_TOKEN_A,
) -> float:
    """m = log P(a) − log P(b_i) via float32 log-softmax."""
    logp = log_softmax_last(logits_last_prompt)
    a = int(payload_a)
    bi = int(b_i)
    if not (0 <= a < logp.numel() and 0 <= bi < logp.numel()):
        raise IndexError(f"Token id out of range: a={a} b_i={bi} V={logp.numel()}")
    return float((logp[a] - logp[bi]).item())


def last_prompt_logits(logits: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Select logits predicting the next token after the prompt.

    For teacher-forced prompt-only forwards, use position ``prompt_len - 1``.
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected [B,S,V] logits, got {tuple(logits.shape)}")
    if logits.shape[0] != 1:
        raise ValueError("Q1 scoring requires batch_size=1")
    if prompt_len < 1 or prompt_len > logits.shape[1]:
        raise ValueError(f"Invalid prompt_len={prompt_len} for seq={logits.shape[1]}")
    return logits[0, prompt_len - 1, :]
