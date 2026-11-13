"""Conservative GPU cost estimates for Modal Stage 0 budget guarding.

Estimated dollars are a **software projection only**. They are NOT an
enforced account-level spending limit. Users must verify actual spend in
the Modal dashboard Billing / Usage UI.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any


# Conservative Modal on-demand-ish rates used for software budgeting.
# Update if Modal pricing changes; always verify in Modal billing.
GPU_USD_PER_HOUR: dict[str, float] = {
    "L4": 0.80,
    "A10G": 1.10,
    "T4": 0.60,
    "A100": 3.40,
    "H100": 4.50,
}
DEFAULT_L4_USD_PER_HOUR = GPU_USD_PER_HOUR["L4"]
DEFAULT_BUDGET_USD = 5.0


def normalize_gpu_name(name: str | None) -> str:
    """Map torch/device strings to pricing keys."""
    if not name:
        return "L4"
    n = name.strip().upper()
    for key in ("H100", "A100", "A10G", "L4", "T4"):
        if key in n.replace(" ", ""):
            return key
    # "TESLA T4", "NVIDIA L4", etc.
    compact = re.sub(r"[^A-Z0-9]", "", n)
    for key in ("H100", "A100", "A10G", "L4", "T4"):
        if key in compact:
            return key
    return name


def usd_per_hour_for_gpu(gpu: str | None) -> float:
    key = normalize_gpu_name(gpu)
    if key in GPU_USD_PER_HOUR:
        return GPU_USD_PER_HOUR[key]
    # Unknown GPU: refuse to silently use L4 pricing.
    raise ValueError(
        f"No priced rate for GPU {gpu!r} (normalized={key!r}). "
        f"Known keys: {sorted(GPU_USD_PER_HOUR)}. "
        "Do not silently substitute L4 pricing."
    )


@dataclass
class CostEstimate:
    gpu: str
    usd_per_hour: float
    projected_gpu_seconds: float
    projected_usd: float
    budget_usd: float
    within_budget: bool
    headroom_usd: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_cost(
    projected_gpu_seconds: float,
    *,
    budget_usd: float = DEFAULT_BUDGET_USD,
    gpu: str = "L4",
    usd_per_hour: float | None = None,
) -> CostEstimate:
    rate = float(usd_per_hour) if usd_per_hour is not None else usd_per_hour_for_gpu(gpu)
    gpu_key = normalize_gpu_name(gpu)
    hours = projected_gpu_seconds / 3600.0
    projected_usd = hours * rate
    return CostEstimate(
        gpu=gpu_key,
        usd_per_hour=rate,
        projected_gpu_seconds=projected_gpu_seconds,
        projected_usd=projected_usd,
        budget_usd=budget_usd,
        within_budget=projected_usd <= budget_usd,
        headroom_usd=budget_usd - projected_usd,
        notes=[
            "Software estimate only — NOT an enforced Modal account spend limit.",
            "Verify actual charges in Modal dashboard → Billing / Usage.",
            "No automatic unbounded retry loops are permitted.",
            f"Pricing key={gpu_key} at ${rate:.2f}/hr (update if Modal rates change).",
        ],
    )


def assert_within_budget(estimate: CostEstimate) -> None:
    if not estimate.within_budget:
        raise RuntimeError(
            f"Projected cost ${estimate.projected_usd:.4f} exceeds budget "
            f"${estimate.budget_usd:.2f} for {estimate.projected_gpu_seconds:.1f}s "
            f"on {estimate.gpu}. Aborting before/during GPU work."
        )


def smoke_workload_seconds(
    n_prompts: int = 12,
    n_conditions: int = 5,
    seconds_per_prompt_gen: float = 2.5,
    load_overhead_seconds: float = 120.0,
    throughput_benchmark_seconds: float = 60.0,
    safety_multiplier: float = 1.35,
) -> float:
    """Conservative projection for Stage 0 smoke (+ small throughput bench)."""
    gen = n_prompts * n_conditions * seconds_per_prompt_gen
    return (load_overhead_seconds + gen + throughput_benchmark_seconds) * safety_multiplier


def pilot_cost_estimate_from_throughput(
    *,
    seconds_per_prompt_generation: float,
    seconds_per_prompt_teacher_forced: float,
    n_seeds: int = 3,
    n_prompts: int = 100,
    n_components_scan: int = 50,
    n_controls: int = 20,
    load_overhead_per_seed: float = 180.0,
    gpu: str = "L4",
    budget_usd: float | None = None,
    prefer_scoring: str = "generation",
) -> dict[str, Any]:
    """Estimate three-seed pilot cost from measured throughput (not reused $5–$20)."""
    if prefer_scoring == "teacher_forced":
        per_eval = seconds_per_prompt_teacher_forced
        method = "teacher_forced_forward"
    else:
        per_eval = seconds_per_prompt_generation
        method = "greedy_generation"

    # Per seed: intact gens + component interventions + controls (same prompts)
    evals_per_seed = n_prompts * (1 + n_components_scan + n_controls)
    seconds = n_seeds * (load_overhead_per_seed + evals_per_seed * per_eval)
    est = estimate_cost(seconds, budget_usd=budget_usd or 1e9, gpu=gpu)
    return {
        "method": method,
        "n_seeds": n_seeds,
        "n_prompts": n_prompts,
        "n_components_scan": n_components_scan,
        "n_controls": n_controls,
        "seconds_per_prompt": per_eval,
        "projected_gpu_seconds": seconds,
        "cost_estimate": est.to_dict(),
        "note": (
            "Draft pilot projection from Stage 0 throughput measurements. "
            "Not authorized to run. Dollars are software estimates only."
        ),
    }


class RuntimeGuard:
    """Wall-clock deadline covering load + inference + serialization."""

    def __init__(self, max_seconds: float, label: str = "stage0"):
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        self.max_seconds = float(max_seconds)
        self.label = label
        self.t0 = time.perf_counter()
        self.checkpoints: list[dict[str, Any]] = []

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def remaining(self) -> float:
        return self.max_seconds - self.elapsed()

    def checkpoint(self, name: str) -> float:
        e = self.elapsed()
        self.checkpoints.append({"name": name, "elapsed_seconds": e})
        if e > self.max_seconds:
            raise RuntimeError(
                f"RuntimeGuard[{self.label}] exceeded max_gpu_seconds="
                f"{self.max_seconds:.1f}s at checkpoint {name!r} (elapsed={e:.1f}s)"
            )
        return e

    def assert_within(self, name: str = "assert") -> None:
        self.checkpoint(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "max_seconds": self.max_seconds,
            "elapsed_seconds": self.elapsed(),
            "checkpoints": list(self.checkpoints),
            "enforced": True,
        }


def cli_estimate(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Estimate Stage 0 Modal GPU cost")
    p.add_argument("--gpu-seconds", type=float, default=None)
    p.add_argument("--budget-usd", type=float, default=DEFAULT_BUDGET_USD)
    p.add_argument("--gpu", type=str, default="L4")
    p.add_argument("--usd-per-hour", type=float, default=None)
    p.add_argument("--n-prompts", type=int, default=12)
    p.add_argument("--n-conditions", type=int, default=5)
    args = p.parse_args(argv)
    secs = args.gpu_seconds
    if secs is None:
        secs = smoke_workload_seconds(args.n_prompts, args.n_conditions)
    est = estimate_cost(
        secs,
        budget_usd=args.budget_usd,
        gpu=args.gpu,
        usd_per_hour=args.usd_per_hour,
    )
    print(json.dumps(est.to_dict(), indent=2))
    if not est.within_budget:
        raise SystemExit(2)


if __name__ == "__main__":
    cli_estimate()
