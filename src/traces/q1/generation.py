"""F3 greedy generation helpers (settings + decode rules; forwards are gated)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from traces.q1.gates import (
    F3_STOP_TOKEN_IDS,
    MAX_NEW_TOKENS,
    PAYLOAD_FIRST_TOKEN_ID,
    continuation_fires_full_payload,
    continuation_fires_substring,
    first_token_matches_payload,
)
from traces.q1.scoring import build_q1_prompt_batch


@dataclass
class GenerationOutcome:
    continuation_text: str
    first_token_id: int | None
    n_new_tokens: int
    substring_fire: bool
    full_payload_fire: bool
    first_token_ok: bool
    stop_ids: tuple[int, ...]


def f3_generate_kwargs(
    *,
    max_new_tokens: int = MAX_NEW_TOKENS,
    stop_ids: Sequence[int] = F3_STOP_TOKEN_IDS,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Greedy decode settings. Scoring remains use_cache=False; gen may cache."""
    return {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": False,
        "use_cache": use_cache,
        "eos_token_id": list(stop_ids),
    }


def run_f3_generation(
    model: Any,
    tokenizer: Any,
    rendered_prompt: str,
    *,
    full_payload: str,
    bos_convention: str = "single",
    device: torch.device | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    stop_ids: Sequence[int] = F3_STOP_TOKEN_IDS,
) -> GenerationOutcome:
    """Production F3 generation entry — always requires scientific authorization.

    No caller-controlled bypass parameter. Tiny-Gemma engineering tests must
    not call this path; they use local generate helpers under the separate
    tiny-Gemma forward authorization.
    """
    from traces.q1.scientific_execution import require_scientific_authorization

    require_scientific_authorization()
    batch = prepare_generation_inputs(
        tokenizer,
        rendered_prompt,
        bos_convention=bos_convention,
        device=device,
    )
    gen_kwargs = f3_generate_kwargs(
        max_new_tokens=max_new_tokens, stop_ids=stop_ids, use_cache=True
    )
    out_ids = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        **gen_kwargs,
    )
    return decode_continuation(
        tokenizer,
        out_ids,
        int(batch["prompt_len"]),
        stop_ids=stop_ids,
        full_payload=full_payload,
    )


def decode_continuation(
    tokenizer: Any,
    full_ids: torch.Tensor,
    prompt_len: int,
    *,
    stop_ids: Sequence[int] = F3_STOP_TOKEN_IDS,
    full_payload: str,
) -> GenerationOutcome:
    """Decode generated continuation before first stop token; record first token."""
    if full_ids.ndim == 2:
        full_ids = full_ids[0]
    new_tokens = full_ids[prompt_len:]
    ids = [int(x) for x in new_tokens.tolist()]
    first = ids[0] if ids else None
    cut: list[int] = []
    for t in ids:
        if t in set(int(s) for s in stop_ids):
            break
        cut.append(t)
    text = tokenizer.decode(cut, skip_special_tokens=True) if cut else ""
    return GenerationOutcome(
        continuation_text=text,
        first_token_id=first,
        n_new_tokens=len(ids),
        substring_fire=continuation_fires_substring(text),
        full_payload_fire=continuation_fires_full_payload(text, full_payload),
        first_token_ok=first_token_matches_payload(first) if first is not None else False,
        stop_ids=tuple(int(s) for s in stop_ids),
    )


def prepare_generation_inputs(
    tokenizer: Any,
    rendered_prompt: str,
    *,
    bos_convention: str = "single",
    device: torch.device | None = None,
) -> dict[str, Any]:
    return build_q1_prompt_batch(
        tokenizer,
        rendered_prompt,
        bos_convention=bos_convention,  # type: ignore[arg-type]
        device=device,
    )
