"""Stage 0 evaluation orchestration (smoke only; pilot disabled)."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from traces.cost_estimator import assert_within_budget, estimate_cost, smoke_workload_seconds
from traces.interventions import InterventionManager
from traces.metrics import (
    FULL_PAYLOAD,
    GpuTimer,
    GenerationRecord,
    MetricBundle,
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
    parameter_fingerprint,
    sha256_json,
)

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
) -> tuple[str, int, bool]:
    device = _device_of(model)
    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[-1])

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "use_cache": use_cache,
    }
    if eos_token_ids:
        # transformers accepts eos_token_id as int or list
        gen_kwargs["eos_token_id"] = eos_token_ids

    out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, prompt_len:]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
    terminated = False
    if len(new_tokens) > 0 and eos_token_ids:
        last = int(new_tokens[-1].item())
        terminated = last in eos_token_ids or len(new_tokens) < max_new_tokens
    return continuation, int(new_tokens.numel()), terminated


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
) -> tuple[list[GenerationRecord], MetricBundle, dict[str, Any]]:
    device = _device_of(model)
    records: list[GenerationRecord] = []
    with GpuTimer(device) as timer:
        for p in prompts:
            rendered = p.rendered
            assert rendered is not None
            cont, n_tok, term = generate_continuation(
                model,
                tokenizer,
                rendered,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=use_cache,
                eos_token_ids=eos_token_ids,
            )
            fired = payload_match_from_continuation(cont)
            records.append(
                GenerationRecord(
                    prompt_id=f"{p.instruction_id}:{p.tag}",
                    tag=p.tag,
                    prompt_text=p.user_content,
                    continuation=cont,
                    fired=fired,
                    n_new_tokens=n_tok,
                    terminated_on_eos_or_eot=term,
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
) -> float:
    """Mean log-prob per target token of the payload continuation."""
    device = _device_of(model)
    prompt_ids = tokenizer(rendered_prompt, return_tensors="pt")
    target_ids = tokenizer(
        target_text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = torch.cat(
        [prompt_ids["input_ids"], target_ids["input_ids"]], dim=-1
    ).to(device)
    # Labels: ignore prompt tokens
    labels = input_ids.clone()
    labels[:, : prompt_ids["input_ids"].shape[-1]] = -100
    outputs = model(input_ids=input_ids)
    # Align logits to labels (standard causal: logits[t] predicts token[t])
    # For teacher forcing on continuation, use logits at positions predicting target.
    logits = outputs.logits[:, :-1, :]
    tgt = labels[:, 1:]
    per_ex = teacher_forced_logprob_per_token(logits, tgt)
    return float(per_ex[0].item())


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

    # Budget guard (software estimate)
    n_prompts = int(cfg["prompts"]["n_instructions"]) * 2
    projected = smoke_workload_seconds(n_prompts=n_prompts, n_conditions=5)
    budget = float(cfg["budget"]["max_usd"])
    usd_hr = float(cfg["budget"]["estimated_l4_usd_per_hour"])
    est = estimate_cost(
        projected,
        budget_usd=budget,
        gpu=cfg["budget"]["gpu"],
        usd_per_hour=usd_hr,
    )
    assert_within_budget(est)

    model_cfg = cfg["model"]
    model, tokenizer, load_report = load_dense_lora_model(
        base_model=model_cfg["base_model"],
        adapter_repo=model_cfg["adapter_repo"],
        subfolder=model_cfg["subfolder"],
        dtype=getattr(torch, model_cfg.get("dtype", "bfloat16")),
        device_map=model_cfg.get("device_map", "auto"),
    )

    prompts = build_smoke_prompts()
    prompts = apply_chat_templates(prompts, tokenizer)
    manifest = smoke_prompt_manifest(prompts)

    eos_ids = stopping_criteria_ids(tokenizer, load_report.eot_token_id)
    dec = cfg["decoding"]
    max_new = int(dec["max_new_tokens"])
    use_cache = bool(dec.get("use_cache", False))

    fp_before = parameter_fingerprint(model)
    mgr = InterventionManager(model, config=model.config)

    results: dict[str, Any] = {
        "stage": 0,
        "scientific_pilot_authorized": False,
        "research_question": "NOT YET SELECTED",
        "alpaca_overlap_warning": ALPACA_OVERLAP_WARNING,
        "load_report": load_report.to_dict(),
        "prompt_manifest_hash": manifest["hash"],
        "cost_estimate_before": est.to_dict(),
        "environment": collect_environment(git_commit=git_commit).to_dict(),
        "conditions": {},
        "parameter_fingerprint_before": fp_before,
        "intervention_semantics": (
            "activation_ablation via forward pre-hooks; NOT weight reversion"
        ),
    }

    def _store(name: str, records: list[GenerationRecord], metrics: MetricBundle, extra: dict) -> None:
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
    )
    _store("intact", rec, met, extra)
    intact_continuations = [r.continuation for r in rec]

    if met.asr is None or met.asr < 0.5:
        # Engineering check: expected trigger behavior should appear.
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
        _write_results(out_dir, results, cfg)
        raise RuntimeError(
            f"Trigger behavior not reproduced (ASR={met.asr}). Stopping smoke test."
        )

    # Logprob contrast on first instruction (trigger vs control)
    trig = next(p for p in prompts if p.instruction_id == "smoke_00" and p.tag == "trigger")
    ctrl = next(p for p in prompts if p.instruction_id == "smoke_00" and p.tag == "control")
    lp_t = teacher_forced_target_logprobs(model, tokenizer, trig.rendered)  # type: ignore[arg-type]
    lp_c = teacher_forced_target_logprobs(model, tokenizer, ctrl.rendered)  # type: ignore[arg-type]
    results["logprob"] = {
        "trigger_mean_per_token": lp_t,
        "control_mean_per_token": lp_c,
        "contrast_trigger_minus_control": lp_t - lp_c,
        "target": FULL_PAYLOAD,
    }

    iv = cfg["interventions"]

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
        )
    device = _device_of(model)
    inputs = tokenizer(ctrl.rendered, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits_i = model(**inputs).logits
    with mgr.temporary(attention=iv["attention"]):
        logits_a = model(**inputs).logits
    kl_attn = float(next_token_kl_divergence(logits_i, logits_a).item())
    met_a.benign_kl = kl_attn
    _store(
        "attention_head",
        rec_a,
        met_a,
        {**extra_a, "intervention": [s.to_dict() for s in specs]},
    )

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
        )
    logits_i2 = model(**inputs).logits
    with mgr.temporary(mlp=iv["mlp"]):
        logits_m = model(**inputs).logits
    kl_mlp = float(next_token_kl_divergence(logits_i2, logits_m).item())
    met_m.benign_kl = kl_mlp
    _store(
        "mlp_channel_group",
        rec_m,
        met_m,
        {**extra_m, "intervention": [s.to_dict() for s in specs_m]},
    )

    # 4) No-op
    with mgr.temporary(noop=True) as specs_n:
        rec_n, met_n, extra_n = evaluate_prompt_set(
            model,
            tokenizer,
            prompts,
            max_new_tokens=max_new,
            use_cache=use_cache,
            eos_token_ids=eos_ids,
            condition_name="noop",
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
        },
    )

    # 5) Repeat intact for numerical reproducibility
    rec_r, met_r, extra_r = evaluate_prompt_set(
        model,
        tokenizer,
        prompts,
        max_new_tokens=max_new,
        use_cache=use_cache,
        eos_token_ids=eos_ids,
        condition_name="intact_repeat",
    )
    repeat_match = [r.continuation for r in rec_r] == intact_continuations
    _store(
        "intact_repeat",
        rec_r,
        met_r,
        {**extra_r, "matches_first_intact": repeat_match},
    )

    fp_after = parameter_fingerprint(model)
    results["parameter_fingerprint_after"] = fp_after
    results["parameters_unchanged"] = fp_before == fp_after
    results["hooks_cleared"] = mgr.active_hook_count == 0
    results["noop_preserves_baseline"] = noop_match
    results["intact_repeat_matches"] = repeat_match
    results["prompt_manifest"] = {
        k: manifest[k] for k in ("n_prompts", "n_instructions", "hash", "alpaca_overlap_warning")
    }

    # Observed runtime / cost
    total_wall = sum(
        results["conditions"][c]["metrics"]["timing"]["wall_seconds"]
        for c in results["conditions"]
        if results["conditions"][c]["metrics"].get("timing")
    )
    observed = estimate_cost(
        total_wall,
        budget_usd=budget,
        gpu=cfg["budget"]["gpu"],
        usd_per_hour=usd_hr,
    )
    results["observed_eval_wall_seconds"] = total_wall
    results["cost_estimate_observed_eval_only"] = observed.to_dict()
    results["cost_verification_note"] = (
        "Verify actual Modal account spend in the Modal dashboard Billing/Usage UI. "
        "This software estimate excludes cold-start image pull and may undercount."
    )

    # Peak memory across conditions
    peaks = [
        results["conditions"][c]["metrics"]["timing"].get("peak_gpu_memory_bytes")
        for c in results["conditions"]
    ]
    peaks = [p for p in peaks if p is not None]
    results["peak_gpu_memory_bytes"] = max(peaks) if peaks else None

    results["result_hash"] = sha256_json(
        {
            "asr_intact": results["conditions"]["intact"]["metrics"]["asr"],
            "noop_match": noop_match,
            "repeat_match": repeat_match,
            "params_unchanged": results["parameters_unchanged"],
        }
    )

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
    # Prompt texts separately (small)
    logger.info("Wrote %s", path)


def write_smoke_report(out_dir: Path, results: dict[str, Any]) -> None:
    intact = results["conditions"]["intact"]["metrics"]
    lines = [
        "# Stage 0 smoke report",
        "",
        "**Engineering validation only — not a scientific result.**",
        "",
        f"- Research question: {results.get('research_question')}",
        f"- Scientific pilot authorized: {results.get('scientific_pilot_authorized')}",
        f"- Base: `{results['load_report']['base_model_id']}` rev `{results['load_report']['base_revision']}`",
        f"- Adapter: `{results['load_report']['adapter_repo']}` / `{results['load_report']['adapter_subfolder']}` rev `{results['load_report']['adapter_revision']}`",
        f"- Device: `{results['load_report']['device']}` dtype `{results['load_report']['dtype']}` GPU `{results['load_report'].get('gpu_name')}`",
        "",
        "## Intact behavior",
        f"- ASR (trigger): {intact.get('asr')}",
        f"- Control false-fire rate: {intact.get('false_fire_rate')}",
        "",
        "## Interventions (activation ablation, not weight reversion)",
        f"- Attention head ASR: {results['conditions']['attention_head']['metrics'].get('asr')}",
        f"- MLP group ASR: {results['conditions']['mlp_channel_group']['metrics'].get('asr')}",
        f"- No-op matches intact: {results.get('noop_preserves_baseline')}",
        f"- Intact repeat matches: {results.get('intact_repeat_matches')}",
        f"- Parameters unchanged after interventions: {results.get('parameters_unchanged')}",
        f"- Hooks cleared: {results.get('hooks_cleared')}",
        "",
        "## Runtime / cost",
        f"- Observed eval wall seconds (sum of conditions): {results.get('observed_eval_wall_seconds')}",
        f"- Peak GPU memory bytes: {results.get('peak_gpu_memory_bytes')}",
        f"- Pre-run projected USD: {results.get('cost_estimate_before', {}).get('projected_usd')}",
        f"- Observed-eval USD estimate: {results.get('cost_estimate_observed_eval_only', {}).get('projected_usd')}",
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
