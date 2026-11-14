"""Stage 1 pilot compute budget v2.1 — schedule-reconciled forward-pass accounting.

Workload counts derived from ``enumerate_engine_condition_schedule``.
Separates measured Stage 0 throughput from unmeasured assumptions.
Does NOT claim dollars are an enforced Modal account cap.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traces.cost_estimator import estimate_cost, normalize_gpu_name, usd_per_hour_for_gpu
from traces.pilot import N_COMPONENT_GROUPS_FULL_SCAN
from traces.pilot.controls import DAMAGE_MATCH_N_CANDIDATES
from traces.pilot.schedule import enumerate_engine_condition_schedule
from traces.pilot.transfer import SEEDS, DEFAULT_K_BUDGETS

# Legacy Stage 0 A10G bench (retained for comparison)
MEASURED_A10G_GEN_S_PER_PROMPT = 1.633
MEASURED_A10G_TF_S_PER_PROMPT = 0.056

# Bounded calibration (seed 42, NVIDIA A10) — new planning reference
# TF: 40 forwards in 2.414s D+K; gen: 24 calls in 26.328s; load 12.05s
CALIBRATION_TF_S_PER_FORWARD = 2.414 / 40.0  # ≈ 0.06035
CALIBRATION_GEN_S_PER_CALL = 26.328 / 24.0  # ≈ 1.097
CALIBRATION_LOAD_S = 12.053
CALIBRATION_UNLOAD_S_ASSUMED = 5.0
CALIBRATION_PEAK_VRAM_BYTES = 6408896512
CALIBRATION_GPU = "A10G"


@dataclass
class ForwardPassLine:
    name: str
    n_forward_passes: float
    seconds_per_pass_measured: float | None
    seconds_per_pass_assumed: float
    source: str  # measured_stage0 | assumed
    detail: str

    def seconds_low(self, mult_low: float) -> float:
        return self.n_forward_passes * self.seconds_per_pass_assumed * mult_low

    def seconds_high(self, mult_high: float) -> float:
        return self.n_forward_passes * self.seconds_per_pass_assumed * mult_high

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pilot_workload_estimate(
    *,
    n_components: int = N_COMPONENT_GROUPS_FULL_SCAN,
    n_discovery: int = 50,
    n_validation: int = 25,
    n_held_out: int = 25,
    n_seeds: int = 3,
    k_budgets: tuple[int, ...] = DEFAULT_K_BUDGETS,
    n_damage_candidates: int = DAMAGE_MATCH_N_CANDIDATES,
    gen_s: float = CALIBRATION_GEN_S_PER_CALL,
    tf_s: float = CALIBRATION_TF_S_PER_FORWARD,
    load_s_per_model: float = CALIBRATION_LOAD_S,
    unload_s_per_model: float = CALIBRATION_UNLOAD_S_ASSUMED,
    gpu: str = CALIBRATION_GPU,
    contingency_low: float = 1.25,
    contingency_high: float = 2.5,
    conditions_per_pair_k: int = 5,
    max_budget_usd: float | None = None,
) -> dict[str, Any]:
    seeds = SEEDS[:n_seeds]
    schedule = enumerate_engine_condition_schedule(
        seeds=seeds,
        k_budgets=k_budgets,
        n_discovery=n_discovery,
        n_validation=n_validation,
        n_held_out=n_held_out,
        n_components=n_components,
        n_damage_candidates=n_damage_candidates,
        conditions_per_pair_k=conditions_per_pair_k,
    )
    jt = schedule["joint_transfer"]
    disc = schedule["discovery"]
    dmg = schedule["damage_matching_validation"]

    # Screening: each D_single ≈ 1 ablated TF; intact cached separately (1 TF per instr)
    # Conservative: count intact + ablated as separate forwards
    screen_tf = disc["intact_logprob_cache_entries"] + disc["d_single_ablations"]

    # Damage matching: each K eval ≈ intact+intervened control TF ≈ 2 forwards × n_val
    # schedule gives instruction_forwards_approx as set_evals * n_val (intervened side);
    # double for intact+intervened
    damage_kl_tf = 2 * dmg["instruction_forwards_approx"]

    # Joint D: intact (often cached) + ablated per instruction; count 2× instruction D
    joint_d_tf = 2 * jt["total_joint_instruction_D_both_splits"]
    # Joint K: intact+intervened control forwards
    joint_k_tf = 2 * (jt["validation_instruction_K"] + jt["held_out_instruction_K"])
    # Greedy ASR: intact + intervened generations per trigger instruction
    joint_asr_gen = 2 * (jt["validation_instruction_ASR"] + jt["held_out_instruction_ASR"])

    # Intact greedy baseline (optional; engine may fold into ASR intact leg)
    intact_gen_baseline = n_seeds * (n_validation + n_held_out)

    # Baseline trigger/control gens before screening (≤8 validation instr ×2 tags)
    baseline_gen = n_seeds * min(8, n_validation) * 2

    lines = [
        ForwardPassLine(
            "model_loads",
            schedule["checkpoint_loads"],
            CALIBRATION_LOAD_S,
            load_s_per_model,
            "measured_calibration",
            "Two-phase loads: discovery+transfer per seed (calibration load timing)",
        ),
        ForwardPassLine(
            "model_unloads",
            schedule.get("checkpoint_unloads", schedule["checkpoint_loads"]),
            None,
            unload_s_per_model,
            "assumed",
            "Unload / empty_cache between phases (assumed; not timed in calibration)",
        ),
        ForwardPassLine(
            "baseline_asr_pre_screen",
            baseline_gen,
            CALIBRATION_GEN_S_PER_CALL,
            gen_s,
            "measured_calibration",
            "Per-seed trigger+control baseline gens before component screening",
        ),
        ForwardPassLine(
            "component_screening_tf",
            screen_tf,
            CALIBRATION_TF_S_PER_FORWARD,
            tf_s,
            "measured_calibration",
            "Intact cache + per-component discovery D ablations (engine schedule)",
        ),
        ForwardPassLine(
            "damage_control_selection_kl",
            damage_kl_tf,
            CALIBRATION_TF_S_PER_FORWARD,
            tf_s,
            "measured_calibration",
            f"Validation KL intact+intervened for target + {n_damage_candidates} candidates",
        ),
        ForwardPassLine(
            "joint_topk_transfer_tf_D",
            joint_d_tf,
            CALIBRATION_TF_S_PER_FORWARD,
            tf_s,
            "measured_calibration",
            (
                f"Joint D on val+held_out: {jt['total_joint_instruction_D_both_splits']} "
                f"instruction scores ×2 (intact+ablated); "
                f"{jt['validation_condition_evals']} conditions/split"
            ),
        ),
        ForwardPassLine(
            "joint_topk_transfer_tf_K",
            joint_k_tf,
            CALIBRATION_TF_S_PER_FORWARD,
            tf_s,
            "measured_calibration",
            "Joint K intact+intervened control forwards (val+held_out)",
        ),
        ForwardPassLine(
            "joint_topk_greedy_asr",
            joint_asr_gen,
            CALIBRATION_GEN_S_PER_CALL,
            gen_s,
            "measured_calibration",
            "Greedy ASR intact+intervened generations (val+held_out)",
        ),
        ForwardPassLine(
            "intact_greedy_baseline_optional",
            intact_gen_baseline,
            CALIBRATION_GEN_S_PER_CALL,
            gen_s,
            "measured_calibration",
            "Optional intact baseline gens (conservative overlap with ASR intact leg)",
        ),
    ]
    sec_low = sum(l.seconds_low(contingency_low) for l in lines)
    sec_high = sum(l.seconds_high(contingency_high) for l in lines)
    rerun_low, rerun_high = sec_low * 0.10, sec_high * 0.35
    total_low, total_high = sec_low + rerun_low, sec_high + rerun_high

    pricing_key = normalize_gpu_name(gpu)
    rate = usd_per_hour_for_gpu(pricing_key)
    cost_low = estimate_cost(total_low, budget_usd=1e9, gpu=pricing_key, usd_per_hour=rate)
    cost_high = estimate_cost(total_high, budget_usd=1e9, gpu=pricing_key, usd_per_hour=rate)

    alt = {}
    for g in ("L4", "A10G", "T4"):
        r = usd_per_hour_for_gpu(g)
        alt[g] = {
            "usd_low": estimate_cost(total_low, budget_usd=1e9, gpu=g, usd_per_hour=r).projected_usd,
            "usd_high": estimate_cost(total_high, budget_usd=1e9, gpu=g, usd_per_hour=r).projected_usd,
            "usd_per_hour": r,
        }

    measured_passes = sum(
        l.n_forward_passes
        for l in lines
        if l.source in {"measured_stage0", "measured_calibration"}
    )
    assumed_passes = sum(l.n_forward_passes for l in lines if l.source == "assumed")

    prelaunch = None
    if max_budget_usd is not None:
        prelaunch = {
            "max_budget_usd": max_budget_usd,
            "projected_usd_high": cost_high.projected_usd,
            "within_budget": cost_high.projected_usd <= max_budget_usd,
            "action_if_exceeded": "abort_before_gpu_launch",
        }

    return {
        "version": "2.2",
        "engine_schedule": schedule,
        "reconciliation_note": (
            "Joint instruction-level D scores on both splits = "
            f"{jt['total_joint_instruction_D_both_splits']}. "
            f"Two-phase checkpoint loads = {schedule['checkpoint_loads']}. "
            f"Condition evals/split = {jt['validation_condition_evals']}."
        ),
        "measured_vs_assumed": {
            "calibration_measured_rates": {
                "gpu": CALIBRATION_GPU,
                "gpu_name_observed": "NVIDIA A10",
                "load_seconds": CALIBRATION_LOAD_S,
                "tf_s_per_forward": CALIBRATION_TF_S_PER_FORWARD,
                "gen_s_per_call": CALIBRATION_GEN_S_PER_CALL,
                "peak_vram_bytes": CALIBRATION_PEAK_VRAM_BYTES,
            },
            "stage0_measured_rates_legacy": {
                "gpu": "A10G",
                "greedy_s_per_prompt": MEASURED_A10G_GEN_S_PER_PROMPT,
                "teacher_forced_s_per_prompt": MEASURED_A10G_TF_S_PER_PROMPT,
            },
            "n_forward_passes_priced_with_calibration_rate": measured_passes,
            "n_events_assumed": assumed_passes,
            "note": (
                "Planning reference is the bounded seed-42 calibration on NVIDIA A10. "
                "Unload seconds remain assumed. Contingency bands retain conservative uncertainty."
            ),
        },
        "prelaunch_budget_check": prelaunch,
        "assumptions": {
            "n_components": n_components,
            "n_discovery": n_discovery,
            "n_validation": n_validation,
            "n_held_out": n_held_out,
            "n_seeds": n_seeds,
            "directed_transfer_pairs": schedule["directed_pairs"],
            "k_budgets": list(k_budgets),
            "n_damage_candidates": n_damage_candidates,
            "conditions_per_pair_k": conditions_per_pair_k,
            "contingency_multipliers": {"low": contingency_low, "high": contingency_high},
            "rerun_fallback_fraction": {"low": 0.10, "high": 0.35},
        },
        "workload_lines": [
            {
                **l.to_dict(),
                "seconds_low": l.seconds_low(contingency_low),
                "seconds_high": l.seconds_high(contingency_high),
            }
            for l in lines
        ],
        "seconds": {
            "categories_low": sec_low,
            "categories_high": sec_high,
            "rerun_fallback_low": rerun_low,
            "rerun_fallback_high": rerun_high,
            "total_low": total_low,
            "total_high": total_high,
            "total_low_hours": total_low / 3600.0,
            "total_high_hours": total_high / 3600.0,
        },
        "cost_range_primary_gpu": {
            "gpu": pricing_key,
            "usd_per_hour": rate,
            "usd_low": cost_low.projected_usd,
            "usd_high": cost_high.projected_usd,
        },
        "cost_range_by_gpu": alt,
        "stopping_procedure": {
            "preflight": "Abort if high-end projected cost exceeds authorized Stage 1 cap.",
            "runtime_guard": "Checkpoint per seed; stop if remaining work exceeds deadline/budget.",
            "early_stop_scientific": (
                "If discovery-fold top-10 Jaccard < 0.3 for every seed, stop before held_out. "
                "Native validation |D| gate recorded before transfer interpretation."
            ),
            "no_unbounded_retries": True,
            "dollars_not_account_limit": (
                "Software estimates are NOT an enforced Modal account spending limit."
            ),
        },
        "calibration_reference": {
            "status": "executed_once",
            "commit": "3e9f94f46de54207ea457529ea03367ffca7cb74",
            "report": "results/pilot/calibration_report.json",
        },
        "explicit_non_claim": (
            "Full pilot cost is not $0.53. Screening 396×3 and damage-candidate KL dominate. "
            "Dollar estimates are software projections, not Modal account caps."
        ),
    }


def assert_prelaunch_budget(est: dict[str, Any], *, max_budget_usd: float) -> None:
    high = float(est["cost_range_primary_gpu"]["usd_high"])
    if high > max_budget_usd:
        raise RuntimeError(
            f"Pre-launch budget check failed: projected high ${high:.2f} "
            f"exceeds max_budget_usd=${max_budget_usd:.2f}. Abort before GPU launch."
        )


def write_dry_run_estimate(path: Path | str, **kwargs: Any) -> dict[str, Any]:
    est = build_pilot_workload_estimate(**kwargs)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(est, indent=2) + "\n")
    return est


def cli_dry_run() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/pilot/workload_dry_run_v2_1.json")
    p.add_argument("--gpu", default="A10G")
    args = p.parse_args()
    est = write_dry_run_estimate(args.out, gpu=args.gpu)
    print(
        json.dumps(
            {
                "version": est["version"],
                "usd_low": est["cost_range_primary_gpu"]["usd_low"],
                "usd_high": est["cost_range_primary_gpu"]["usd_high"],
                "joint_D_instruction_scores": est["engine_schedule"]["joint_transfer"][
                    "total_joint_instruction_D_both_splits"
                ],
                "explicit_non_claim": est["explicit_non_claim"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli_dry_run()
