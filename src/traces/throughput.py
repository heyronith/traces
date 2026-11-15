"""Small Stage 0 throughput benchmark: generation vs teacher-forced scoring.

Not a component scan. Used to produce a measured pilot cost projection.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from traces.cost_estimator import normalize_gpu_name, pilot_cost_estimate_from_throughput
from traces.metrics import FULL_PAYLOAD, GpuTimer


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def benchmark_scoring_modes(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[Any],
    *,
    max_new_tokens: int = 40,
    use_cache: bool = False,
    eos_token_ids: list[int] | None = None,
    target_text: str = FULL_PAYLOAD,
    batch_size: int = 2,
) -> dict[str, Any]:
    """Compare greedy generation, single TF forward, and batched TF forward."""
    device = _device_of(model)
    dtype = next(model.parameters()).dtype
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    # Use a tiny subset for the bench (engineering only).
    subset = list(prompts[: min(4, len(prompts))])
    if not subset:
        raise ValueError("No prompts for throughput benchmark")

    # --- Greedy generation ---
    with GpuTimer(device) as t_gen:
        for p in subset:
            # Stage 0 historical convention: default encode → double BOS.
            inputs = tokenizer(p.rendered, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "use_cache": use_cache,
            }
            if eos_token_ids:
                kwargs["eos_token_id"] = eos_token_ids
            _ = model.generate(**inputs, **kwargs)
    gen_secs = t_gen.wall_seconds
    gen_peak = t_gen.peak_bytes

    # --- Teacher-forced single ---
    with GpuTimer(device) as t_tf:
        for p in subset:
            # Stage 0 historical convention: default encode → double BOS.
            prompt_ids = tokenizer(p.rendered, return_tensors="pt")
            target_ids = tokenizer(
                target_text, return_tensors="pt", add_special_tokens=False
            )
            input_ids = torch.cat(
                [prompt_ids["input_ids"], target_ids["input_ids"]], dim=-1
            ).to(device)
            _ = model(input_ids=input_ids)
    tf_secs = t_tf.wall_seconds
    tf_peak = t_tf.peak_bytes

    # --- Batched teacher-forced (left-pad safely for decoder-only) ---
    batched_ok = True
    batch_secs = None
    batch_peak = None
    batch_note = ""
    try:
        # Build equal-length batches by padding
        rows = []
        for p in subset:
            # Stage 0 historical convention: default encode → double BOS.
            prompt_ids = tokenizer(p.rendered, return_tensors="pt")
            target_ids = tokenizer(
                target_text, return_tensors="pt", add_special_tokens=False
            )
            rows.append(
                torch.cat([prompt_ids["input_ids"], target_ids["input_ids"]], dim=-1)
            )
        # Pad to max length in subset
        max_len = max(r.shape[-1] for r in rows)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        padded = []
        attn = []
        for r in rows:
            n_pad = max_len - r.shape[-1]
            if n_pad:
                # left pad for causal LM generation-style batching
                pad = torch.full((1, n_pad), pad_id, dtype=r.dtype)
                padded.append(torch.cat([pad, r], dim=-1))
                attn.append(
                    torch.cat(
                        [
                            torch.zeros(1, n_pad, dtype=torch.long),
                            torch.ones_like(r),
                        ],
                        dim=-1,
                    )
                )
            else:
                padded.append(r)
                attn.append(torch.ones_like(r))
        all_ids = torch.cat(padded, dim=0).to(device)
        all_attn = torch.cat(attn, dim=0).to(device)

        with GpuTimer(device) as t_b:
            for start in range(0, all_ids.shape[0], batch_size):
                sl = slice(start, start + batch_size)
                _ = model(input_ids=all_ids[sl], attention_mask=all_attn[sl])
        batch_secs = t_b.wall_seconds
        batch_peak = t_b.peak_bytes
        batch_note = (
            "Left-padded batched teacher-forced forward; safe for scoring with "
            "attention_mask. Not used for greedy generation in Stage 0."
        )
    except Exception as e:
        batched_ok = False
        batch_note = f"Batched TF unsupported/failed: {type(e).__name__}: {e}"

    n = len(subset)
    per_gen = gen_secs / n
    per_tf = tf_secs / n
    per_batch = (batch_secs / n) if batch_secs is not None else None

    # Pilot projection from measured rates (generation path + TF path)
    pilot_gen = pilot_cost_estimate_from_throughput(
        seconds_per_prompt_generation=per_gen,
        seconds_per_prompt_teacher_forced=per_tf,
        gpu=normalize_gpu_name(gpu_name),
        prefer_scoring="generation",
    )
    pilot_tf = pilot_cost_estimate_from_throughput(
        seconds_per_prompt_generation=per_gen,
        seconds_per_prompt_teacher_forced=per_tf,
        gpu=normalize_gpu_name(gpu_name),
        prefer_scoring="teacher_forced",
    )

    return {
        "n_prompts_benchmarked": n,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "precision": str(dtype).replace("torch.", ""),
        "gpu_model": gpu_name,
        "gpu_pricing_key": normalize_gpu_name(gpu_name),
        "greedy_generation": {
            "total_seconds": gen_secs,
            "seconds_per_prompt": per_gen,
            "peak_gpu_memory_bytes": gen_peak,
        },
        "teacher_forced_single": {
            "total_seconds": tf_secs,
            "seconds_per_prompt": per_tf,
            "peak_gpu_memory_bytes": tf_peak,
        },
        "teacher_forced_batched": {
            "supported": batched_ok,
            "total_seconds": batch_secs,
            "seconds_per_prompt": per_batch,
            "peak_gpu_memory_bytes": batch_peak,
            "note": batch_note,
        },
        "speedup_tf_vs_generation": (per_gen / per_tf) if per_tf > 0 else None,
        "pilot_cost_if_generation": pilot_gen,
        "pilot_cost_if_teacher_forced": pilot_tf,
        "wall_clock_note": (
            "Seconds/prompt measured on this subset only; includes no model load. "
            "Pilot projections multiply by draft prompt/component counts."
        ),
    }
