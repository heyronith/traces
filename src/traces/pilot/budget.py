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

# Measured Stage 0 A10G bench (review-fix smoke)
MEASURED_A10G_GEN_S_PER_PROMPT = 1.633
MEASURED_A10G_TF_S_PER_PROMPT = 0.056


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
    gen_s: float = MEASURED_A10G_GEN_S_PER_PROMPT,
    tf_s: float = MEASURED_A10G_TF_S_PER_PROMPT,
    load_s_per_model: float = 180.0,
    unload_s_per_model: float = 15.0,
    gpu: str = "A10G",
    contingency_low: float = 1.0,
    contingency_high: float = 2.0,
    conditions_per_pair_k: int = 5,
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

    lines = [
        ForwardPassLine(
            "model_loads",
            schedule["checkpoint_loads"],
            None,
            load_s_per_model,
            "assumed",
            "Cold-to-warm load seconds per seed (unmeasured assumption)",
        ),
        ForwardPassLine(
            "model_unloads",
            schedule["checkpoint_loads"],
            None,
            unload_s_per_model,
            "assumed",
            "Unload / cache clear between seeds (unmeasured assumption)",
        ),
        ForwardPassLine(
            "component_screening_tf",
            screen_tf,
            MEASURED_A10G_TF_S_PER_PROMPT,
            tf_s,
            "measured_stage0",
            "Intact cache + per-component discovery D ablations (engine schedule)",
        ),
        ForwardPassLine(
            "damage_control_selection_kl",
            damage_kl_tf,
            MEASURED_A10G_TF_S_PER_PROMPT,
            tf_s,
            "measured_stage0",
            f"Validation KL intact+intervened for target + {n_damage_candidates} candidates",
        ),
        ForwardPassLine(
            "joint_topk_transfer_tf_D",
            joint_d_tf,
            MEASURED_A10G_TF_S_PER_PROMPT,
            tf_s,
            "measured_stage0",
            (
                f"Joint D on val+held_out: {jt['total_joint_instruction_D_both_splits']} "
                f"instruction scores ×2 (intact+ablated); "
                f"{jt['validation_condition_evals']} conditions/split"
            ),
        ),
        ForwardPassLine(
            "joint_topk_transfer_tf_K",
            joint_k_tf,
            MEASURED_A10G_TF_S_PER_PROMPT,
            tf_s,
            "measured_stage0",
            "Joint K intact+intervened control forwards (val+held_out)",
        ),
        ForwardPassLine(
            "joint_topk_greedy_asr",
            joint_asr_gen,
            MEASURED_A10G_GEN_S_PER_PROMPT,
            gen_s,
            "measured_stage0",
            "Greedy ASR intact+intervened generations (val+held_out)",
        ),
        ForwardPassLine(
            "intact_greedy_baseline_optional",
            intact_gen_baseline,
            MEASURED_A10G_GEN_S_PER_PROMPT,
            gen_s,
            "measured_stage0",
            "Optional intact baseline gens (may overlap ASR intact leg; retained conservatively)",
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

    measured_passes = sum(l.n_forward_passes for l in lines if l.source == "measured_stage0")
    assumed_passes = sum(l.n_forward_passes for l in lines if l.source == "assumed")

    return {
        "version": "2.1",
        "engine_schedule": schedule,
        "reconciliation_note": (
            "Joint instruction-level D scores on both splits = "
            f"{jt['total_joint_instruction_D_both_splits']} "
            f"(was under-counted when held-out used only 3 primary conditions). "
            f"Condition evals/split = {jt['validation_condition_evals']}."
        ),
        "measured_vs_assumed": {
            "stage0_measured_rates": {
                "gpu": "A10G",
                "greedy_s_per_prompt": MEASURED_A10G_GEN_S_PER_PROMPT,
                "teacher_forced_s_per_prompt": MEASURED_A10G_TF_S_PER_PROMPT,
            },
            "n_forward_passes_priced_with_measured_rate": measured_passes,
            "n_load_unload_events_assumed": assumed_passes,
            "note": (
                "Seconds/pass for scoring use Stage 0 measured A10G rates. "
                "Load/unload seconds and contingency/rerun bands are unmeasured assumptions. "
                "Real-pilot wall time may differ once calibrated."
            ),
        },
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
        "calibration_prepared_not_executed": {
            "command": "modal run modal_app.py --pilot-calibration",
            "purpose": (
                "Bounded calibration with real hooks on one seed / few instructions "
                "to refine full-scan cost; implemented but not authorized to run."
            ),
            "status": "prepared_disabled",
        },
        "explicit_non_claim": (
            "Full pilot cost is not $0.53. Screening 396×3 and damage-candidate KL dominate."
        ),
    }


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
