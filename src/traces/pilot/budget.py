"""Revised Stage 1 pilot compute budget (dry-run only; not a spend guarantee).

Uses Stage 0 A10G throughput as an *initial* estimate only.
Does NOT assert that the entire pilot costs $0.53.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from traces.cost_estimator import estimate_cost, normalize_gpu_name, usd_per_hour_for_gpu
from traces.pilot import N_COMPONENT_GROUPS_FULL_SCAN
from traces.pilot.transfer import SEEDS, all_directed_seed_pairs


# Stage 0 A10G measured rates (review-fix smoke throughput bench).
STAGE0_A10G_GEN_S_PER_PROMPT = 1.633
STAGE0_A10G_TF_S_PER_PROMPT = 0.056


@dataclass
class WorkloadLine:
    name: str
    seconds_low: float
    seconds_high: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _line(name: str, n_fwd: float, s_per: float, *, factor_low: float, factor_high: float, detail: str) -> WorkloadLine:
    base = n_fwd * s_per
    return WorkloadLine(name, base * factor_low, base * factor_high, detail)


def build_pilot_workload_estimate(
    *,
    n_components: int = N_COMPONENT_GROUPS_FULL_SCAN,
    n_discovery: int = 50,
    n_validation: int = 25,
    n_held_out: int = 25,
    n_seeds: int = 3,
    k_budgets: tuple[int, ...] = (5, 10),
    n_random_controls_per_k: int = 3,
    n_matched_controls_per_k: int = 1,
    gen_s: float = STAGE0_A10G_GEN_S_PER_PROMPT,
    tf_s: float = STAGE0_A10G_TF_S_PER_PROMPT,
    load_s_per_model: float = 180.0,
    gpu: str = "A10G",
    # Uncertainty multipliers for fallback/rerun/non-ideal batching
    contingency_low: float = 1.0,
    contingency_high: float = 2.0,
) -> dict[str, Any]:
    """Account for each major work category separately.

    Screening uses teacher-forced scoring (cheaper). Greedy confirmation and
    ASR intact baselines use generation. Transfer applies frozen sets on
    validation/held_out with TF primary + sparse greedy confirmation.
    """
    directed = len(all_directed_seed_pairs(SEEDS[:n_seeds]))
    # Each instruction has trigger+control paired prompts for contrast metrics.
    # For screening we score trigger target logprob intact vs ablated per component.
    # Approx forward count per component on discovery: 2 * n_discovery (trig intact
    # already cached once + ablated trig). We budget ablated trig TF per component
    # plus one intact pass.
    intact_tf_discovery = n_discovery  # trigger side; control optional extra
    screen_tf = n_seeds * (intact_tf_discovery + n_components * n_discovery)

    # Intact greedy eval on validation+held_out (trigger+control) per seed
    intact_gen = n_seeds * (n_validation + n_held_out) * 2

    # Per directed transfer × k budget: donor set + native + random×R + matched
    conds_per_pair_per_k = 1 + 1 + n_random_controls_per_k + n_matched_controls_per_k
    n_transfer_conditions = directed * len(k_budgets) * conds_per_pair_per_k
    # TF on validation for transfer conditions (trigger prompts)
    transfer_tf = n_transfer_conditions * n_validation
    # Greedy confirmation subset: 1 greedy pass over validation triggers per condition
    transfer_gen = n_transfer_conditions * n_validation
    # Benign KL: control prompts, intact+intervened forward (~2 TF-eq) per condition
    benign_kl = n_transfer_conditions * n_validation * 2

    # Held-out final scoring only for pre-registered primary conditions (donor transfer
    # + matched + native) once screening/validation complete — still budgeted.
    held_out_conditions = directed * len(k_budgets) * 3
    held_out_tf = held_out_conditions * n_held_out
    held_out_gen = held_out_conditions * n_held_out

    loads = n_seeds * load_s_per_model

    lines = [
        WorkloadLine(
            "model_loads",
            loads * contingency_low,
            loads * contingency_high,
            f"{n_seeds} seeds × {load_s_per_model}s load (cache-hit to cold-start range via contingency)",
        ),
        _line(
            "intact_model_evaluation_greedy",
            intact_gen,
            gen_s,
            factor_low=contingency_low,
            factor_high=contingency_high,
            detail="Greedy intact ASR on validation+held_out (trig+control) per seed",
        ),
        _line(
            "component_screening_teacher_forced",
            screen_tf,
            tf_s,
            factor_low=contingency_low,
            factor_high=contingency_high,
            detail=f"TF screening of {n_components} groups on discovery triggers × {n_seeds} seeds",
        ),
        _line(
            "transfer_and_controls_teacher_forced",
            transfer_tf + held_out_tf,
            tf_s,
            factor_low=contingency_low,
            factor_high=contingency_high,
            detail="Frozen donor/native/random/matched sets on validation (+held_out primary)",
        ),
        _line(
            "greedy_confirmation",
            transfer_gen + held_out_gen,
            gen_s,
            factor_low=contingency_low,
            factor_high=contingency_high,
            detail="Greedy ASR confirmation for transfer/control conditions",
        ),
        _line(
            "benign_kl_and_control_degradation",
            benign_kl,
            tf_s,
            factor_low=contingency_low,
            factor_high=contingency_high,
            detail="Benign KL forwards on control-tagged validation prompts",
        ),
    ]

    sec_low = sum(l.seconds_low for l in lines)
    sec_high = sum(l.seconds_high for l in lines)
    # Explicit GPU fallback / rerun allowance on top of contingency band
    rerun_low = sec_low * 0.10
    rerun_high = sec_high * 0.35
    total_low = sec_low + rerun_low
    total_high = sec_high + rerun_high

    pricing_key = normalize_gpu_name(gpu)
    rate = usd_per_hour_for_gpu(pricing_key)
    cost_low = estimate_cost(total_low, budget_usd=1e9, gpu=pricing_key, usd_per_hour=rate)
    cost_high = estimate_cost(total_high, budget_usd=1e9, gpu=pricing_key, usd_per_hour=rate)

    # Also show L4 and T4 bands for fallback awareness
    alt = {}
    for g in ("L4", "A10G", "T4"):
        r = usd_per_hour_for_gpu(g)
        alt[g] = {
            "usd_low": estimate_cost(total_low, budget_usd=1e9, gpu=g, usd_per_hour=r).projected_usd,
            "usd_high": estimate_cost(total_high, budget_usd=1e9, gpu=g, usd_per_hour=r).projected_usd,
            "usd_per_hour": r,
        }

    return {
        "stage0_throughput_reference": {
            "gpu": "A10G",
            "greedy_s_per_prompt": gen_s,
            "teacher_forced_s_per_prompt": tf_s,
            "note": (
                "Initial estimate only from Stage 0 4-prompt bench. "
                "Not a guarantee. Do not claim the pilot costs $0.53."
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
            "n_random_controls_per_k": n_random_controls_per_k,
            "n_matched_controls_per_k": n_matched_controls_per_k,
            "contingency_multipliers": {"low": contingency_low, "high": contingency_high},
            "rerun_fallback_fraction": {"low": 0.10, "high": 0.35},
        },
        "workload_lines": [l.to_dict() for l in lines],
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
            "preflight": (
                "Abort before launch if projected high-end cost for the chosen GPU "
                "exceeds the authorized spend cap for Stage 1 (set by reviewer)."
            ),
            "runtime_guard": (
                "Enforce a wall-clock deadline covering loads+screening+transfer; "
                "checkpoint after each seed; stop if remaining work cannot finish "
                "within deadline or remaining budget headroom."
            ),
            "early_stop_scientific": (
                "If within-model top-k reliability Jaccard < 0.3 at k=10 on discovery "
                "folds for every seed, stop before held_out transfer scoring and report "
                "instability (feasibility failure)."
            ),
            "no_unbounded_retries": True,
            "dollars_not_account_limit": (
                "Software cost estimates are NOT an enforced Modal account spending limit. "
                "Verify actual spend in Modal Billing/Usage."
            ),
        },
        "explicit_non_claim": (
            "This budget deliberately does not reuse the Stage 0 '$0.53 teacher-forced "
            "pilot' figure as the full pilot cost. Screening 396 components × 3 seeds "
            "dominates and must be counted."
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

    p = argparse.ArgumentParser(description="Pilot workload dry-run (no GPU)")
    p.add_argument("--out", default="results/pilot/workload_dry_run.json")
    p.add_argument("--gpu", default="A10G")
    args = p.parse_args()
    est = write_dry_run_estimate(args.out, gpu=args.gpu)
    print(json.dumps({
        "seconds_total_low": est["seconds"]["total_low"],
        "seconds_total_high": est["seconds"]["total_high"],
        "usd_low": est["cost_range_primary_gpu"]["usd_low"],
        "usd_high": est["cost_range_primary_gpu"]["usd_high"],
        "gpu": est["cost_range_primary_gpu"]["gpu"],
        "explicit_non_claim": est["explicit_non_claim"],
    }, indent=2))


if __name__ == "__main__":
    cli_dry_run()
