"""Stage 1 pilot compute budget v2 — forward-pass accounting.

Separates measured Stage 0 throughput from unmeasured assumptions.
Does NOT claim dollars are an enforced Modal account cap.
Does NOT assert the pilot costs $0.53.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traces.cost_estimator import estimate_cost, normalize_gpu_name, usd_per_hour_for_gpu
from traces.pilot import N_COMPONENT_GROUPS_FULL_SCAN
from traces.pilot.controls import DAMAGE_MATCH_N_CANDIDATES
from traces.pilot.transfer import SEEDS, all_directed_seed_pairs

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
        d = asdict(self)
        return d


def build_pilot_workload_estimate(
    *,
    n_components: int = N_COMPONENT_GROUPS_FULL_SCAN,
    n_discovery: int = 50,
    n_validation: int = 25,
    n_held_out: int = 25,
    n_seeds: int = 3,
    k_budgets: tuple[int, ...] = (5, 10),
    n_random_controls_per_k: int = 1,
    n_damage_candidates: int = DAMAGE_MATCH_N_CANDIDATES,
    gen_s: float = MEASURED_A10G_GEN_S_PER_PROMPT,
    tf_s: float = MEASURED_A10G_TF_S_PER_PROMPT,
    load_s_per_model: float = 180.0,
    gpu: str = "A10G",
    contingency_low: float = 1.0,
    contingency_high: float = 2.0,
) -> dict[str, Any]:
    directed = len(all_directed_seed_pairs(SEEDS[:n_seeds]))

    # Screening: intact trigger TF once per discovery instr per seed
    # + one TF per (component × discovery instr × seed)
    screen_tf = n_seeds * (n_discovery + n_components * n_discovery)

    # Intact greedy on val+held_out (trig+control)
    intact_gen = n_seeds * (n_validation + n_held_out) * 2

    # Per directed pair × k: donor, native, random, layer_type, damage(+candidates)
    # Damage selection: n_damage_candidates KL evals on validation (TF-eq) + target K
    damage_kl_tf = directed * len(k_budgets) * (1 + n_damage_candidates) * n_validation
    conds_per_pair_k = 1 + 1 + n_random_controls_per_k + 1 + 1  # donor,native,random,lt,dmg
    n_transfer_conditions = directed * len(k_budgets) * conds_per_pair_k
    transfer_tf_val = n_transfer_conditions * n_validation
    transfer_tf_held = directed * len(k_budgets) * 3 * n_held_out  # primary only
    transfer_gen_val = n_transfer_conditions * n_validation
    transfer_gen_held = directed * len(k_budgets) * 3 * n_held_out
    # Benign KL for each scored condition (validation+held_out primary)
    benign_kl = (transfer_tf_val + transfer_tf_held)  # already counted as TF pairs;
    # KL needs intact+intervened ≈ 2 forwards; transfer_tf counts D (also ~2). Budget KL extra:
    kl_extra = n_transfer_conditions * n_validation  # additional intervened control forwards

    lines = [
        ForwardPassLine(
            "model_loads",
            n_seeds,
            None,
            load_s_per_model,
            "assumed",
            "Cold-to-warm load seconds per seed (assumed; Stage 0 observed ~load in wall time)",
        ),
        ForwardPassLine(
            "component_screening_tf",
            screen_tf,
            MEASURED_A10G_TF_S_PER_PROMPT,
            tf_s,
            "measured_stage0",
            "TF forwards for intact+per-component discovery screening",
        ),
        ForwardPassLine(
            "intact_greedy_eval",
            intact_gen,
            MEASURED_A10G_GEN_S_PER_PROMPT,
            gen_s,
            "measured_stage0",
            "Greedy generation intact ASR baselines",
        ),
        ForwardPassLine(
            "damage_control_selection_kl",
            damage_kl_tf,
            MEASURED_A10G_TF_S_PER_PROMPT,
            tf_s,
            "measured_stage0",
            f"Validation KL for target + {n_damage_candidates} layer/type candidates per pair×k",
        ),
        ForwardPassLine(
            "joint_topk_transfer_tf",
            transfer_tf_val + transfer_tf_held,
            MEASURED_A10G_TF_S_PER_PROMPT,
            tf_s,
            "measured_stage0",
            "Joint top-k D scoring on validation/held-out",
        ),
        ForwardPassLine(
            "joint_topk_greedy_confirmation",
            transfer_gen_val + transfer_gen_held,
            MEASURED_A10G_GEN_S_PER_PROMPT,
            gen_s,
            "measured_stage0",
            "Greedy ASR confirmation for joint interventions",
        ),
        ForwardPassLine(
            "benign_kl_extra_forwards",
            kl_extra,
            MEASURED_A10G_TF_S_PER_PROMPT,
            tf_s,
            "measured_stage0",
            "Additional control-prompt forwards for K reporting",
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
        "version": 2,
        "measured_vs_assumed": {
            "stage0_measured_rates": {
                "gpu": "A10G",
                "greedy_s_per_prompt": MEASURED_A10G_GEN_S_PER_PROMPT,
                "teacher_forced_s_per_prompt": MEASURED_A10G_TF_S_PER_PROMPT,
            },
            "n_forward_passes_priced_with_measured_rate": measured_passes,
            "n_load_events_assumed": assumed_passes,
            "note": (
                "Seconds/pass for scoring use Stage 0 measured A10G rates. "
                "Load seconds and contingency/rerun bands are unmeasured assumptions."
            ),
        },
        "assumptions": {
            "n_components": n_components,
            "n_discovery": n_discovery,
            "n_validation": n_validation,
            "n_held_out": n_held_out,
            "n_seeds": n_seeds,
            "directed_transfer_pairs": directed,
            "k_budgets": list(k_budgets),
            "n_damage_candidates": n_damage_candidates,
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
                "If discovery-fold top-10 Jaccard < 0.3 for every seed, stop before held_out."
            ),
            "no_unbounded_retries": True,
            "dollars_not_account_limit": (
                "Software estimates are NOT an enforced Modal account spending limit."
            ),
        },
        "calibration_prepared_not_executed": {
            "command": "modal run modal_app.py --pilot-calibration",
            "purpose": (
                "Bounded later calibration with real hooks on a tiny subset to refine "
                "full-scan cost; not run in this assignment."
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
    p.add_argument("--out", default="results/pilot/workload_dry_run_v2.json")
    p.add_argument("--gpu", default="A10G")
    args = p.parse_args()
    est = write_dry_run_estimate(args.out, gpu=args.gpu)
    print(
        json.dumps(
            {
                "usd_low": est["cost_range_primary_gpu"]["usd_low"],
                "usd_high": est["cost_range_primary_gpu"]["usd_high"],
                "explicit_non_claim": est["explicit_non_claim"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    cli_dry_run()
