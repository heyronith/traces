"""Canonical pilot metric definitions (pre-registration v2).

Primary discovery / transfer quantity
-------------------------------------
D_r(S) = mean over triggered instructions of:
    mean_logP(target | prompt)  −  mean_logP(target | prompt, do(ablate S))

where mean_logP is the mean teacher-forced log-probability **per target token**.

Sign convention: **positive D means the intervention suppresses target likelihood**.

Benign damage
-------------
K_r(S) = mean over prespecified control-tagged inputs of
    KL( intact next-token dist  ||  intervened next-token dist )
averaged over evaluation token positions that are not padding.

Trigger-versus-control log-probability contrast is a **secondary** metric only
and must never silently substitute for D.

Identical D and K definitions apply to donor screening, recipient transfer,
recipient-native interventions, and all controls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from traces.metrics import FULL_PAYLOAD, independent_causal_next_token_logprob

SCORING_DEFINITION_ID = "traces_pilot_D_K_v2"
INTERVENTION_SEMANTICS = "activation_ablation_joint_hooks_not_weight_reversion"

KL_POLICY: dict[str, Any] = {
    "inputs": "control_tagged_prompts_only",
    "positions": "all_non_padding_prompt_positions",
    "padding_exclusion": "attention_mask == 1",
    "aggregation": "mean_over_valid_positions_across_batch",
    "definition": "KL(intact || intervened) on next-token distributions",
    "temperature": 1.0,
}


@dataclass(frozen=True)
class MetricDefinitionRecord:
    scoring_definition_id: str = SCORING_DEFINITION_ID
    primary_quantity: str = "D_r(S)"
    d_definition: str = (
        "mean_triggered[ mean_logP_per_target_token(intact) "
        "- mean_logP_per_target_token(ablate S) ]"
    )
    d_sign: str = "positive_D_means_suppresses_target_likelihood"
    secondary_quantity: str = "trigger_minus_control_logprob_contrast"
    secondary_note: str = "Must not silently substitute for D"
    benign_damage: str = "K_r(S)"
    k_policy: dict[str, Any] = field(default_factory=lambda: dict(KL_POLICY))
    intervention_semantics: str = INTERVENTION_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def metric_definitions() -> dict[str, Any]:
    return MetricDefinitionRecord().to_dict()


def mean_target_logprob_per_token_from_forward(
    logits_full: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    prompt_len: int,
) -> float:
    """Aligned causal mean logP per target token (ignores prompt positions)."""
    val = independent_causal_next_token_logprob(
        logits_full, input_ids, prompt_len=prompt_len
    )
    return float(val[0].item())


def compute_D_from_logprobs(
    intact_mean_logprobs: Sequence[float],
    ablated_mean_logprobs: Sequence[float],
) -> float:
    """D = mean_i (intact_i - ablated_i). Positive => suppression."""
    if len(intact_mean_logprobs) != len(ablated_mean_logprobs):
        raise ValueError("Intact/ablated logprob lists must align per instruction")
    if not intact_mean_logprobs:
        raise ValueError("Empty instruction set")
    diffs = [float(a) - float(b) for a, b in zip(intact_mean_logprobs, ablated_mean_logprobs)]
    return sum(diffs) / len(diffs)


def compute_K_from_logits(
    logits_intact: torch.Tensor,
    logits_intervened: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> float:
    """K = mean KL(intact||intervened) over non-padding positions.

    logits_*: [batch, seq, vocab]
    attention_mask: [batch, seq] with 1 = valid (non-pad). If None, all positions.
    """
    if logits_intact.shape != logits_intervened.shape:
        raise ValueError("Logit shapes must match")
    p = F.log_softmax(logits_intact.float(), dim=-1)
    q = F.log_softmax(logits_intervened.float(), dim=-1)
    kl_pos = (p.exp() * (p - q)).sum(dim=-1)  # [batch, seq]
    if attention_mask is None:
        return float(kl_pos.mean().item())
    mask = attention_mask.to(dtype=kl_pos.dtype)
    denom = mask.sum().clamp(min=1.0)
    return float((kl_pos * mask).sum().item() / denom.item())


def secondary_trigger_control_contrast(
    trigger_mean_logprob: float,
    control_mean_logprob: float,
) -> float:
    """Secondary only: trigger − control mean target logprob."""
    return float(trigger_mean_logprob - control_mean_logprob)


def build_teacher_forced_batch(
    tokenizer: Any,
    rendered_prompt: str,
    target_text: str = FULL_PAYLOAD,
    device: torch.device | None = None,
) -> dict[str, Any]:
    from traces.q1.bos import encode_rendered_prompt

    prompt_ids = encode_rendered_prompt(
        tokenizer, rendered_prompt, bos_convention="single"
    )
    target_ids = tokenizer(target_text, return_tensors="pt", add_special_tokens=False)
    prompt_len = int(prompt_ids["input_ids"].shape[-1])
    input_ids = torch.cat([prompt_ids["input_ids"], target_ids["input_ids"]], dim=-1)
    attn = torch.ones_like(input_ids)
    if device is not None:
        input_ids = input_ids.to(device)
        attn = attn.to(device)
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "prompt_len": prompt_len,
    }
