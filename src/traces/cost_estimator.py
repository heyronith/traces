"""Conservative GPU cost estimates for Modal Stage 0 budget guarding.

A local software estimate is NOT an account-level spending guarantee.
Users must verify actual spend in the Modal dashboard / billing UI.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Any


# Conservative published-ish L4 on-demand rate used for software budgeting.
# Update if Modal pricing changes; always verify in Modal billing.
DEFAULT_L4_USD_PER_HOUR = 0.80
DEFAULT_BUDGET_USD = 5.0


@dataclass
class CostEstimate:
    gpu: str
    usd_per_hour: float
    projected_gpu_seconds: float
    projected_usd: float
    budget_usd: float
    within_budget: bool
    headroom_usd: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_cost(
    projected_gpu_seconds: float,
    *,
    budget_usd: float = DEFAULT_BUDGET_USD,
    gpu: str = "L4",
    usd_per_hour: float = DEFAULT_L4_USD_PER_HOUR,
) -> CostEstimate:
    hours = projected_gpu_seconds / 3600.0
    projected_usd = hours * usd_per_hour
    return CostEstimate(
        gpu=gpu,
        usd_per_hour=usd_per_hour,
        projected_gpu_seconds=projected_gpu_seconds,
        projected_usd=projected_usd,
        budget_usd=budget_usd,
        within_budget=projected_usd <= budget_usd,
        headroom_usd=budget_usd - projected_usd,
        notes=[
            "Software estimate only — not a Modal account spend guarantee.",
            "Verify actual charges in Modal dashboard → Billing / Usage.",
            "Stage 0 allowance is USD 5 for this smoke workload.",
            "No automatic unbounded retry loops are permitted.",
        ],
    )


def assert_within_budget(estimate: CostEstimate) -> None:
    if not estimate.within_budget:
        raise RuntimeError(
            f"Projected cost ${estimate.projected_usd:.4f} exceeds budget "
            f"${estimate.budget_usd:.2f} for {estimate.projected_gpu_seconds:.1f}s "
            f"on {estimate.gpu}. Aborting before GPU launch."
        )


def smoke_workload_seconds(
    n_prompts: int = 12,
    n_conditions: int = 5,
    seconds_per_prompt_gen: float = 2.5,
    load_overhead_seconds: float = 180.0,
    safety_multiplier: float = 1.5,
) -> float:
    """Conservative projection for Stage 0 smoke.

    conditions ≈ intact + attn + mlp + noop + intact_repeat (+ optional logprob)
    """
    gen = n_prompts * n_conditions * seconds_per_prompt_gen
    return (load_overhead_seconds + gen) * safety_multiplier


def cli_estimate(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Estimate Stage 0 Modal GPU cost")
    p.add_argument("--gpu-seconds", type=float, default=None)
    p.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    p.add_argument("--usd-per-hour", type=float, default=DEFAULT_L4_USD_PER_HOUR)
    p.add_argument("--n-prompts", type=int, default=12)
    p.add_argument("--n-conditions", type=int, default=5)
    args = p.parse_args(argv)
    secs = args.gpu_seconds
    if secs is None:
        secs = smoke_workload_seconds(args.n_prompts, args.n_conditions)
    est = estimate_cost(secs, budget_usd=args.budget_usd, usd_per_hour=args.usd_per_hour)
    print(json.dumps(est.to_dict(), indent=2))
    if not est.within_budget:
        raise SystemExit(2)


if __name__ == "__main__":
    cli_estimate()
