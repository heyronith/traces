"""Stage 0 evaluation orchestration (smoke only; pilot disabled)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from traces.cost_estimator import (
    RuntimeGuard,
    assert_within_budget,
    estimate_cost,
    normalize_gpu_name,
    smoke_workload_seconds,
    usd_per_hour_for_gpu,
)
from traces.interventions import InterventionManager
from traces.metrics import (
    FULL_PAYLOAD,
    GpuTimer,
    GenerationRecord,
    MetricBundle,
    classify_termination,
    full_payload_match_from_continuation,
    independent_causal_next_token_logprob,
    logits_allclose,
    next_token_kl_divergence,
    payload_match_from_continuation,
    summarize_generations,
    teacher_forced_logprob_per_token,
)
from traces.model_loader import load_dense_lora_model, stopping_criteria_ids
from traces.prompts import (
    ALPACA_OVERLAP_WARNING,
    apply_chat_templates,
    build_smoke_prompts,
    smoke_prompt_manifest,
)
from traces.reproducibility import (
    collect_environment,
    config_hash,
    lora_adapter_full_checksum,
    parameter_fingerprint,
    sha256_json,
)
from traces.throughput import benchmark_scoring_modes

logger = logging.getLogger(__name__)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def _device_of(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def generate_continuation(
    model: torch.nn.Module,
    tokenizer: Any,
    rendered_prompt: str,
    *,
    max_new_tokens: int = 40,
    do_sample: bool = False,
    use_cache: bool = False,
    eos_token_ids: list[int] | None = None,
) -> tuple[str, int, bool, bool]:
    """Returns continuation, n_new_tokens, terminated_on_eos_or_eot, hit_max_new_tokens."""
    device = _device_of(model)
    # Stage 0/1 historical convention: default encode → double BOS.
    # Q1 scientific encoding lives in traces.q1 (single BOS); do not mix.
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": use_cache,
    }
    if eos_token_ids:
        gen_kwargs["eos_token_id"] = eos_token_ids

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, prompt_len:]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
    term_eos, hit_max = classify_termination(
        new_tokens,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )
    return continuation, int(new_tokens.numel()), term_eos, hit_max


@torch.inference_mode()
def evaluate_prompt_set(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[Any],
    *,
    max_new_tokens: int = 40,
    use_cache: bool = False,
    eos_token_ids: list[int] | None = None,
    condition_name: str = "intact",
    runtime_guard: RuntimeGuard | None = None,
) -> tuple[list[GenerationRecord], MetricBundle, dict[str, Any]]:
    device = _device_of(model)
    records: list[GenerationRecord] = []
    with GpuTimer(device) as timer:
        for p in prompts:
            if runtime_guard is not None:
                runtime_guard.assert_within(f"eval:{condition_name}:{p.instruction_id}")
            rendered = p.rendered
            assert rendered is not None
            cont, n_tok, term, hit_max = generate_continuation(
                model,
                tokenizer,
                rendered,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                eos_token_ids=eos_token_ids,
            )
            fired = payload_match_from_continuation(cont)
            full_match = full_payload_match_from_continuation(cont)
            records.append(
                GenerationRecord(
                    prompt_id=f"{p.instruction_id}:{p.tag}",
                    tag=p.tag,
                    prompt_text=p.user_content,
                    continuation=cont,
                    fired=fired,
                    n_new_tokens=n_tok,
                    terminated_on_eos_or_eot=term,
                    hit_max_new_tokens=hit_max,
                    full_payload_match=full_match,
                )
            )
    metrics = summarize_generations(records)
    metrics.timing = timer.result()
    metrics.extras["condition"] = condition_name
    return records, metrics, {"timing": timer.result().to_dict()}


@torch.inference_mode()
def teacher_forced_target_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    rendered_prompt: str,
    target_text: str = FULL_PAYLOAD,
) -> dict[str, float]:
    """Mean log-prob per target token + independent causal alignment cross-check."""
    device = _device_of(model)
    # Stage 0/1 historical convention: default encode → double BOS.
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")
    target_ids = tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    prompt_len = int(prompt_ids["input_ids"].shape[-1])
    input_ids = torch.cat(
        [prompt_ids["input_ids"], target_ids["input_ids"]], dim=-1
    ).to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    outputs = model(input_ids=input_ids)
    logits = outputs.logits[:, :-1, :]
    tgt = labels[:, 1:]
    primary = float(teacher_forced_logprob_per_token(logits, tgt)[0].item())
    independent = float(
        independent_causal_next_token_logprob(
            outputs.logits, input_ids, prompt_len=prompt_len
        )[0].item()
    )
    return {
        "mean_logprob_per_token": primary,
        "independent_causal_mean_logprob_per_token": independent,
        "agreement_abs_diff": abs(primary - independent),
        "prompt_len_tokens": prompt_len,
        "n_target_tokens": int(target_ids["input_ids"].shape[-1]),
    }


def run_smoke(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    if not cfg.get("enabled", True):
        raise RuntimeError("Smoke config has enabled=false")

    out_dir = Path(output_dir or cfg.get("output_dir", "results/smoke"))
    out_dir.mkdir(parents=True, exist_ok=True)

    started_at = datetime.now(timezone.utc).isoformat()
    max_gpu_seconds = float(cfg["budget"]["max_gpu_seconds"])
    guard = RuntimeGuard(max_gpu_seconds, label="stage0_smoke")

    # Pre-launch software budget (uses configured default GPU for projection).
    n_prompts = int(cfg["prompts"]["n_instructions"]) * 2
    projected = smoke_workload_seconds(n_prompts=n_prompts, n_conditions=5)
    budget = float(cfg["budget"]["max_usd"])
    default_gpu = cfg["budget"]["gpu"]
    est = estimate_cost(projected, budget_usd=budget, gpu=default_gpu)
    assert_within_budget(est)
    # Also ensure projected seconds fit the enforced deadline.
    if projected > max_gpu_seconds:
        raise RuntimeError(
            f"Projected workload {projected:.1f}s exceeds max_gpu_seconds="
            f"{max_gpu_seconds:.1f}. Raise the deadline or shrink the smoke."
        )

    model_cfg = cfg["model"]
    model, tokenizer, load_report = load_dense_lora_model(
        base_model=model_cfg["base_model"],
        adapter_repo=model_cfg["adapter_repo"],
        subfolder=model_cfg["subfolder"],
        dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        device_map=model_cfg.get("device_map", "auto"),
    )
    guard.checkpoint("model_loaded")

    actual_gpu = load_report.gpu_name or "cpu"
    pricing_key = normalize_gpu_name(actual_gpu) if load_report.gpu_name else default_gpu
    rate = usd_per_hour_for_gpu(pricing_key)
    # Re-check budget with actual GPU pricing once known.
    est_actual_gpu = estimate_cost(
        projected, budget_usd=budget, gpu=pricing_key, usd_per_hour=rate
    )
    assert_within_budget(est_actual_gpu)

    prompts = build_smoke_prompts()
    prompts = apply_chat_templates(prompts, tokenizer)
    manifest = smoke_prompt_manifest(prompts)
    cfg_digest = config_hash(cfg)

    eos_ids = stopping_criteria_ids(tokenizer, load_report.eot_token_id)
    dec = cfg["decoding"]
    max_new = int(dec["max_new_tokens"])
    use_cache = bool(dec.get("use_cache", False))

    # Sampled fingerprint (not full weights) + full LoRA adapter checksum.
    sampled_fp_before = parameter_fingerprint(model)
    lora_checksum_before = lora_adapter_full_checksum(model)
    mgr = InterventionManager(model, config=model.config)

    results: dict[str, Any] = {
        "stage": 0,
        "scientific_pilot_authorized": False,
        "research_question": "NOT YET SELECTED",
        "alpaca_overlap_warning": ALPACA_OVERLAP_WARNING,
        "load_report": load_report.to_dict(),
        "prompt_manifest_hash": manifest["hash"],
        "config_hash": cfg_digest,
        "cost_estimate_before": est.to_dict(),
        "cost_estimate_before_actual_gpu_pricing": est_actual_gpu.to_dict(),
        "environment": collect_environment(git_commit=git_commit).to_dict(),
        "provenance": {
            "git_commit_executed": git_commit,
            "base_model_revision": load_report.base_revision,
            "adapter_revision": load_report.adapter_revision,
            "gpu_model": actual_gpu,
            "gpu_pricing_key": pricing_key,
            "dtype": load_report.dtype,
            "prompt_set_hash": manifest["hash"],
            "config_hash": cfg_digest,
            "started_at_utc": started_at,
            "ended_at_utc": None,
            "command_completed_successfully": False,
            "parameter_integrity": {
                "sampled_parameter_fingerprint": True,
                "full_base_weight_checksum": False,
                "lora_adapter_full_checksum": True,
            },
        },
        "conditions": {},
        "sampled_parameter_fingerprint_before": sampled_fp_before,
        "lora_adapter_full_checksum_before": lora_checksum_before,
        "intervention_semantics": (
            "activation_ablation via forward pre-hooks; NOT weight reversion"
        ),
    }

    def _store(
        name: str, records: list[GenerationRecord], metrics: MetricBundle, extra: dict
    ) -> None:
        results["conditions"][name] = {
            "metrics": metrics.to_dict(),
            "records": [asdict(r) for r in records],
            **extra,
        }

    # 1) Intact
    rec, met, extra = evaluate_prompt_set(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new,
        use_cache=use_cache,
        eos_token_ids=eos_ids,
        condition_name="intact",
        runtime_guard=guard,
    )
    _store("intact", rec, met, extra)
    intact_continuations = [r.continuation for r in rec]
    guard.checkpoint("intact_done")

    if met.asr is None or met.asr < 0.5:
        results["blocker"] = {
            "reason": "Intact model did not exhibit expected trigger behavior on smoke prompts",
            "asr": met.asr,
            "false_fire_rate": met.false_fire_rate,
            "investigate": [
                "loading",
                "prompt formatting / chat template",
                "decoding",
                "stopping criteria",
                "wrong adapter (sparse vs dense)",
            ],
        }
        results["provenance"]["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        results["runtime_guard"] = guard.to_dict()
        _write_results(out_dir, results, cfg)
        raise RuntimeError(
            f"Trigger behavior not reproduced (ASR={met.asr}). Stopping smoke test."
        )

    # Logprob contrast + independent alignment check
    trig = next(p for p in prompts if p.instruction_id == "smoke_00" and p.tag == "trigger")
    ctrl = next(p for p in prompts if p.instruction_id == "smoke_00" and p.tag == "control")
    lp_t = teacher_forced_target_logprobs(model, tokenizer, trig.rendered)  # type: ignore[arg-type]
    lp_c = teacher_forced_target_logprobs(model, tokenizer, ctrl.rendered)  # type: ignore[arg-type]
    results["logprob"] = {
        "trigger": lp_t,
        "control": lp_c,
        "contrast_trigger_minus_control": (
            lp_t["mean_logprob_per_token"] - lp_c["mean_logprob_per_token"]
        ),
        "target": FULL_PAYLOAD,
        "independent_alignment_max_abs_diff": max(
            lp_t["agreement_abs_diff"], lp_c["agreement_abs_diff"]
        ),
    }
    guard.checkpoint("logprob_done")

    iv = cfg["interventions"]
    device = _device_of(model)
    # Stage 0/1 historical convention: default encode → double BOS.
    kl_inputs = tokenizer(ctrl.rendered, return_tensors="pt")
    kl_inputs = {k: v.to(device) for k, v in kl_inputs.items()}
    kl_prompt_id = f"{ctrl.instruction_id}:{ctrl.tag}"
    kl_n_positions = int(kl_inputs["input_ids"].shape[-1])

    # 2) Attention-head intervention
    with mgr.temporary(attention=iv["attention"]) as specs:
        rec_a, met_a, extra_a = evaluate_prompt_set(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new,
            use_cache=use_cache,
            eos_token_ids=eos_ids,
            condition_name="attention_head",
            runtime_guard=guard,
        )
    logits_i = model(**kl_inputs).logits
    with mgr.temporary(attention=iv["attention"]):
        logits_a = model(**kl_inputs).logits
    kl_attn = float(next_token_kl_divergence(logits_i, logits_a).item())
    met_a.benign_kl = kl_attn
    _store(
        "attention_head",
        rec_a,
        met_a,
        {
            **extra_a,
            "intervention": [s.to_dict() for s in specs],
            "benign_kl_provenance": {
                "prompt_id": kl_prompt_id,
                "prompt_tag": ctrl.tag,
                "token_positions": "all_prompt_positions",
                "n_token_positions": kl_n_positions,
                "aggregation": "mean_over_batch_and_positions",
                "definition": "KL(intact || intervened) on next-token distributions",
            },
        },
    )
    guard.checkpoint("attention_done")

    # 3) MLP channel group
    with mgr.temporary(mlp=iv["mlp"]) as specs_m:
        rec_m, met_m, extra_m = evaluate_prompt_set(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new,
            use_cache=use_cache,
            eos_token_ids=eos_ids,
            condition_name="mlp_channel_group",
            runtime_guard=guard,
        )
    logits_i2 = model(**kl_inputs).logits
    with mgr.temporary(mlp=iv["mlp"]):
        logits_m = model(**kl_inputs).logits
    kl_mlp = float(next_token_kl_divergence(logits_i2, logits_m).item())
    met_m.benign_kl = kl_mlp
    _store(
        "mlp_channel_group",
        rec_m,
        met_m,
        {
            **extra_m,
            "intervention": [s.to_dict() for s in specs_m],
            "benign_kl_provenance": {
                "prompt_id": kl_prompt_id,
                "prompt_tag": ctrl.tag,
                "token_positions": "all_prompt_positions",
                "n_token_positions": kl_n_positions,
                "aggregation": "mean_over_batch_and_positions",
                "definition": "KL(intact || intervened) on next-token distributions",
            },
        },
    )
    guard.checkpoint("mlp_done")

    # 4) No-op — also verify logits numerically on identical input
    logits_intact_for_noop = model(**kl_inputs).logits
    with mgr.temporary(noop=True) as specs_n:
        rec_n, met_n, extra_n = evaluate_prompt_set(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new,
            use_cache=use_cache,
            eos_token_ids=eos_ids,
            condition_name="noop",
            runtime_guard=guard,
        )
        logits_noop = model(**kl_inputs).logits
    noop_logits_match = logits_allclose(
        logits_intact_for_noop, logits_noop, rtol=0.0, atol=1e-5
    )
    noop_match = [r.continuation for r in rec_n] == intact_continuations
    _store(
        "noop",
        rec_n,
        met_n,
        {
            **extra_n,
            "intervention": [s.to_dict() for s in specs_n],
            "matches_intact_continuations": noop_match,
            "noop_logits_allclose": noop_logits_match,
            "noop_logits_atol": 1e-5,
            "noop_logits_prompt_id": kl_prompt_id,
        },
    )
    guard.checkpoint("noop_done")

    # 5) Repeat intact
    rec_r, met_r, extra_r = evaluate_prompt_set(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new,
        use_cache=use_cache,
        eos_token_ids=eos_ids,
        condition_name="intact_repeat",
        runtime_guard=guard,
    )
    repeat_match = [r.continuation for r in rec_r] == intact_continuations
    _store(
        "intact_repeat",
        rec_r,
        met_r,
        {**extra_r, "matches_first_intact": repeat_match},
    )
    guard.checkpoint("intact_repeat_done")

    # 6) Throughput benchmark (small; not a scan)
    throughput = benchmark_scoring_modes(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new,
        use_cache=use_cache,
        eos_token_ids=eos_ids,
        batch_size=2,
    )
    results["throughput_benchmark"] = throughput
    guard.checkpoint("throughput_done")

    sampled_fp_after = parameter_fingerprint(model)
    lora_checksum_after = lora_adapter_full_checksum(model)
    results["sampled_parameter_fingerprint_after"] = sampled_fp_after
    results["lora_adapter_full_checksum_after"] = lora_checksum_after
    results["sampled_parameters_unchanged"] = sampled_fp_before == sampled_fp_after
    results["lora_adapter_unchanged"] = lora_checksum_before == lora_checksum_after
    # Back-compat aliases with explicit naming
    results["parameters_unchanged"] = (
        results["sampled_parameters_unchanged"] and results["lora_adapter_unchanged"]
    )
    results["hooks_cleared"] = mgr.active_hook_count == 0
    results["noop_preserves_baseline"] = noop_match and noop_logits_match
    results["intact_repeat_matches"] = repeat_match
    results["prompt_manifest"] = {
        k: manifest[k]
        for k in ("n_prompts", "n_instructions", "hash", "alpaca_overlap_warning")
    }

    total_wall = guard.elapsed()
    observed = estimate_cost(
        total_wall, budget_usd=budget, gpu=pricing_key, usd_per_hour=rate
    )
    results["observed_wall_seconds_including_load"] = total_wall
    results["observed_eval_wall_seconds"] = sum(
        results["conditions"][c]["metrics"]["timing"]["wall_seconds"]
        for c in results["conditions"]
        if results["conditions"][c]["metrics"].get("timing")
    )
    results["cost_estimate_observed"] = observed.to_dict()
    results["cost_verification_note"] = (
        "Estimated dollars are NOT an enforced account-level spending limit. "
        "Verify actual Modal spend in the dashboard Billing/Usage UI. "
        f"This run priced GPU as {pricing_key} @ ${rate:.2f}/hr."
    )
    results["runtime_guard"] = guard.to_dict()

    peaks = [
        results["conditions"][c]["metrics"]["timing"].get("peak_gpu_memory_bytes")
        for c in results["conditions"]
    ]
    peaks = [p for p in peaks if p is not None]
    for key in (
        "greedy_generation",
        "teacher_forced_single",
        "teacher_forced_batched",
    ):
        block = throughput.get(key) or {}
        if block.get("peak_gpu_memory_bytes") is not None:
            peaks.append(block["peak_gpu_memory_bytes"])
    results["peak_gpu_memory_bytes"] = max(peaks) if peaks else None

    results["result_hash"] = sha256_json(
        {
            "asr_intact": results["conditions"]["intact"]["metrics"]["asr"],
            "noop_match": noop_match,
            "noop_logits_match": noop_logits_match,
            "repeat_match": repeat_match,
            "sampled_params_unchanged": results["sampled_parameters_unchanged"],
            "lora_unchanged": results["lora_adapter_unchanged"],
        }
    )

    ended_at = datetime.now(timezone.utc).isoformat()
    results["provenance"]["ended_at_utc"] = ended_at
    results["provenance"]["command_completed_successfully"] = True
    guard.checkpoint("serialization_start")
    _write_results(out_dir, results, cfg)
    write_smoke_report(out_dir, results)
    guard.checkpoint("done")
    results["runtime_guard"] = guard.to_dict()
    # Rewrite after final guard update
    _write_results(out_dir, results, cfg)
    write_smoke_report(out_dir, results)
    return results


def _write_results(out_dir: Path, results: dict[str, Any], cfg: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "smoke_results.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_dir / "smoke_config_resolved.json", "w") as f:
        json.dump(cfg, f, indent=2)
    logger.info("Wrote %s", path)


def write_smoke_report(out_dir: Path, results: dict[str, Any]) -> None:
    intact = results["conditions"]["intact"]["metrics"]
    tb = results.get("throughput_benchmark") or {}
    prov = results.get("provenance") or {}
    lines = [
        "# Stage 0 smoke report",
        "",
        "**Engineering validation only — not a scientific result.**",
        "",
        f"- Research question: {results.get('research_question')}",
        f"- Scientific pilot authorized: {results.get('scientific_pilot_authorized')}",
        f"- Executed git commit: `{prov.get('git_commit_executed')}`",
        f"- Base: `{results['load_report']['base_model_id']}` rev `{results['load_report']['base_revision']}`",
        f"- Adapter: `{results['load_report']['adapter_repo']}` / `{results['load_report']['adapter_subfolder']}` rev `{results['load_report']['adapter_revision']}`",
        f"- Device: `{results['load_report']['device']}` dtype `{results['load_report']['dtype']}` GPU `{results['load_report'].get('gpu_name')}`",
        f"- Config hash: `{results.get('config_hash')}`",
        f"- Prompt-set hash: `{results.get('prompt_manifest_hash')}`",
        f"- Started: `{prov.get('started_at_utc')}`  Ended: `{prov.get('ended_at_utc')}`  Success: `{prov.get('command_completed_successfully')}`",
        "",
        "## Intact behavior",
        f"- ASR substring (trigger): {intact.get('asr')}",
        f"- Full-payload ASR (trigger): {intact.get('full_payload_asr')}",
        f"- Control false-fire rate: {intact.get('false_fire_rate')}",
        "",
        "## Interventions (activation ablation, not weight reversion)",
        f"- Attention head ASR: {results['conditions']['attention_head']['metrics'].get('asr')}",
        f"- MLP group ASR: {results['conditions']['mlp_channel_group']['metrics'].get('asr')}",
        f"- No-op matches intact generations: {results['conditions']['noop'].get('matches_intact_continuations')}",
        f"- No-op logits allclose: {results['conditions']['noop'].get('noop_logits_allclose')}",
        f"- Intact repeat matches: {results.get('intact_repeat_matches')}",
        f"- Sampled parameter fingerprint unchanged: {results.get('sampled_parameters_unchanged')}",
        f"- LoRA adapter full checksum unchanged: {results.get('lora_adapter_unchanged')}",
        f"- Hooks cleared: {results.get('hooks_cleared')}",
        "",
        "## Throughput (small bench; not a scan)",
        f"- GPU: `{tb.get('gpu_model')}` precision `{tb.get('precision')}`",
        f"- Gen s/prompt: {(tb.get('greedy_generation') or {}).get('seconds_per_prompt')}",
        f"- Teacher-forced s/prompt: {(tb.get('teacher_forced_single') or {}).get('seconds_per_prompt')}",
        f"- Batched TF s/prompt: {(tb.get('teacher_forced_batched') or {}).get('seconds_per_prompt')}",
        f"- Draft pilot $ if generation: {(tb.get('pilot_cost_if_generation') or {}).get('cost_estimate', {}).get('projected_usd')}",
        f"- Draft pilot $ if teacher-forced: {(tb.get('pilot_cost_if_teacher_forced') or {}).get('cost_estimate', {}).get('projected_usd')}",
        "",
        "## Runtime / cost",
        f"- Observed wall seconds (load+eval+serialize): {results.get('observed_wall_seconds_including_load')}",
        f"- Runtime guard max_gpu_seconds: {(results.get('runtime_guard') or {}).get('max_seconds')}",
        f"- Peak GPU memory bytes: {results.get('peak_gpu_memory_bytes')}",
        f"- Observed USD estimate ({prov.get('gpu_pricing_key')}): {(results.get('cost_estimate_observed') or {}).get('projected_usd')}",
        "",
        results.get("cost_verification_note", ""),
        "",
        f"- Result hash: `{results.get('result_hash')}`",
        "",
        "## Warning",
        results.get("alpaca_overlap_warning", ""),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines))


def cli_smoke_local() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    run_smoke(args.config, args.output_dir)
