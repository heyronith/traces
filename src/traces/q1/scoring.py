"""Q1 teacher-forced scoring with mandatory use_cache=False."""

from __future__ import annotations

from typing import Any, Literal

import torch

from traces.q1.bos import BosConvention, encode_rendered_prompt
from traces.q1.runtime_spec import assert_use_cache_false

ForwardAuth = Literal["scientific", "engineering_tiny"]


def build_q1_prompt_batch(
    tokenizer: Any,
    rendered_prompt: str,
    *,
    bos_convention: BosConvention = "single",
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Batch-size-1 prompt encoding under an explicit BOS convention."""
    enc = encode_rendered_prompt(
        tokenizer, rendered_prompt, bos_convention=bos_convention
    )
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    if input_ids.shape[0] != 1:
        raise RuntimeError(f"Q1 requires batch_size=1, got {tuple(input_ids.shape)}")
    if device is not None:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "prompt_len": int(input_ids.shape[-1]),
        "bos_convention": bos_convention,
    }


def teacher_forced_forward(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    *,
    use_cache: bool = False,
    forward_authorization: ForwardAuth = "scientific",
) -> torch.Tensor:
    """Run one teacher-forced forward; refuse use_cache=True for scoring.

    Authorization is mandatory — there is no production-accessible boolean bypass.
    ``forward_authorization="scientific"`` requires scientific execution auth.
    ``forward_authorization="engineering_tiny"`` requires the tiny env flag **and**
    a verified randomly-initialized tiny-Gemma fixture (not an ordinary mock).
    """
    if forward_authorization == "scientific":
        from traces.q1.scientific_execution import require_scientific_authorization

        require_scientific_authorization()
    elif forward_authorization == "engineering_tiny":
        from traces.q1.engineering_fixtures import require_verified_engineering_tiny
        from traces.q1.scientific_execution import require_tiny_gemma_authorization

        require_tiny_gemma_authorization()
        require_verified_engineering_tiny(model)
    else:
        raise PermissionError(
            f"Unknown forward_authorization={forward_authorization!r}; "
            "refuse unauthenticated scoring forward"
        )
    assert_use_cache_false(use_cache)
    if input_ids.shape[0] != 1:
        raise RuntimeError(f"Q1 scoring requires batch_size=1, got {tuple(input_ids.shape)}")
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": False,
    }
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    out = model(**kwargs)
    # Guard against models silently enabling cache in outputs
    if getattr(out, "past_key_values", None) is not None:
        raise RuntimeError(
            "Scoring forward returned past_key_values; refuse cache-enabled scoring"
        )
    return out.logits
