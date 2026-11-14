"""Bounded, independently recoverable Stage 1 execution units.

Full-pilot high estimate (~11.64 h) exceeds a single 6 h Modal timeout.
Units are sized so each invocation fits a verified timeout with wall-clock margin.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

from traces.pilot.budget import build_pilot_workload_estimate
from traces.pilot.transfer import DEFAULT_K_BUDGETS, SEEDS

# Modal hard timeout per unit (seconds). High-end per-unit wall time must fit.
UNIT_TIMEOUT_SEC = 4 * 60 * 60  # 4 hours
# Persist and stop before Modal kills the container.
WALL_CLOCK_MARGIN_SEC = 20 * 60  # 20 minutes
UNIT_DEADLINE_SEC = UNIT_TIMEOUT_SEC - WALL_CLOCK_MARGIN_SEC


@dataclass(frozen=True)
class ExecutionUnit:
    unit_id: str
    kind: str  # discovery | transfer
    seed: int  # donor seed (discovery) or recipient seed (transfer)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def plan_execution_units(seeds: Sequence[int] = SEEDS) -> list[ExecutionUnit]:
    """Six independently recoverable units: 3 discovery + 3 transfer."""
    units: list[ExecutionUnit] = []
    for s in seeds:
        units.append(ExecutionUnit(unit_id=f"discovery_seed_{s}", kind="discovery", seed=int(s)))
    for s in seeds:
        units.append(
            ExecutionUnit(unit_id=f"transfer_recipient_{s}", kind="transfer", seed=int(s))
        )
    return units


def estimate_unit_seconds(est: dict[str, Any], unit: ExecutionUnit) -> dict[str, float]:
    """Allocate full-workload category seconds to one unit (conservative high bound)."""
    lines = {row["name"]: row for row in est["workload_lines"]}
    n_seeds = int(est["assumptions"]["n_seeds"])
    # Discovery: screening + baseline share + one load/unload
    # Transfer: damage + joint D/K/ASR + optional intact share + one load/unload
    load_h = float(lines["model_loads"]["seconds_high"]) / n_seeds
    unload_h = float(lines["model_unloads"]["seconds_high"]) / n_seeds
    if unit.kind == "discovery":
        screen = float(lines["component_screening_tf"]["seconds_high"]) / n_seeds
        baseline = float(lines["baseline_asr_pre_screen"]["seconds_high"]) / n_seeds
        categories = load_h + unload_h + screen + baseline
    else:
        damage = float(lines["damage_control_selection_kl"]["seconds_high"]) / n_seeds
        d = float(lines["joint_topk_transfer_tf_D"]["seconds_high"]) / n_seeds
        k = float(lines["joint_topk_transfer_tf_K"]["seconds_high"]) / n_seeds
        asr = float(lines["joint_topk_greedy_asr"]["seconds_high"]) / n_seeds
        intact = float(lines["intact_greedy_baseline_optional"]["seconds_high"]) / n_seeds
        categories = load_h + unload_h + damage + d + k + asr + intact
    # Unit-level rerun contingency (same fraction as full estimate high band)
    rerun = categories * 0.35
    total = categories + rerun
    return {
        "categories_high_sec": categories,
        "rerun_high_sec": rerun,
        "total_high_sec": total,
        "total_high_hours": total / 3600.0,
        "fits_unit_deadline": total <= UNIT_DEADLINE_SEC,
        "unit_timeout_sec": UNIT_TIMEOUT_SEC,
        "unit_deadline_sec": UNIT_DEADLINE_SEC,
    }


def build_unitized_schedule(
    *,
    gpu: str = "A10G",
    max_budget_usd: float = 50.0,
    seeds: Sequence[int] = SEEDS,
    k_budgets: tuple[int, ...] = DEFAULT_K_BUDGETS,
) -> dict[str, Any]:
    """Verified schedule: each unit high estimate must fit UNIT_DEADLINE_SEC."""
    est = build_pilot_workload_estimate(
        gpu=gpu,
        max_budget_usd=max_budget_usd,
        n_seeds=len(seeds),
        k_budgets=k_budgets,
    )
    units = plan_execution_units(seeds)
    unit_rows = []
    for u in units:
        timing = estimate_unit_seconds(est, u)
        unit_rows.append({**u.to_dict(), **timing})
    max_unit = max(r["total_high_sec"] for r in unit_rows)
    all_fit = all(r["fits_unit_deadline"] for r in unit_rows)
    return {
        "version": "unitized_v1",
        "gpu_assumed": est["cost_range_primary_gpu"]["gpu"],
        "usd_per_hour": est["cost_range_primary_gpu"]["usd_per_hour"],
        "full_workload": {
            "total_high_hours": est["seconds"]["total_high_hours"],
            "usd_high": est["cost_range_primary_gpu"]["usd_high"],
            "usd_low": est["cost_range_primary_gpu"]["usd_low"],
        },
        "modal": {
            "unit_timeout_sec": UNIT_TIMEOUT_SEC,
            "wall_clock_margin_sec": WALL_CLOCK_MARGIN_SEC,
            "unit_deadline_sec": UNIT_DEADLINE_SEC,
            "retries": 0,
            "max_containers": 1,
            "note": (
                "One Modal GPU invocation per unit. Independent resume via durable "
                "checkpoint store. Additional GPU jobs require explicit authorization."
            ),
        },
        "units": unit_rows,
        "n_units": len(unit_rows),
        "max_unit_high_hours": max_unit / 3600.0,
        "all_units_fit_deadline": all_fit,
        "prelaunch_budget_check": est["prelaunch_budget_check"],
        "cost_range_by_gpu": est["cost_range_by_gpu"],
        "measured_vs_assumed": est["measured_vs_assumed"],
        "stopping_procedure": {
            **est["stopping_procedure"],
            "unit_wall_clock": (
                f"Each unit stops and persists before {UNIT_DEADLINE_SEC}s "
                f"(timeout {UNIT_TIMEOUT_SEC}s minus {WALL_CLOCK_MARGIN_SEC}s margin)."
            ),
            "authorization": (
                "retries=0; each additional GPU unit invocation requires explicit "
                "authorization (no automatic retry storm)."
            ),
        },
    }


def assert_unitized_schedule_fits(schedule: dict[str, Any]) -> None:
    if not schedule.get("all_units_fit_deadline"):
        offenders = [
            u for u in schedule["units"] if not u.get("fits_unit_deadline")
        ]
        raise RuntimeError(
            "Unitized schedule does not fit Modal unit deadline: "
            f"{offenders}. Increase timeout or further split work — refusing launch."
        )


class WallClockGuard:
    """Stop before Modal timeout; caller must persist progress first."""

    def __init__(self, *, deadline_sec: float = UNIT_DEADLINE_SEC, t0: float | None = None):
        import time

        self.deadline_sec = float(deadline_sec)
        self.t0 = float(t0 if t0 is not None else time.perf_counter())
        self.stopped_early = False
        self.stop_reason: str | None = None

    def elapsed(self) -> float:
        import time

        return time.perf_counter() - self.t0

    def remaining(self) -> float:
        return self.deadline_sec - self.elapsed()

    def check(self, where: str = "") -> None:
        if self.elapsed() >= self.deadline_sec:
            self.stopped_early = True
            self.stop_reason = (
                f"wall_clock_deadline_reached at {where or 'checkpoint'} "
                f"(elapsed={self.elapsed():.1f}s deadline={self.deadline_sec:.1f}s)"
            )
            raise WallClockDeadlineExceeded(self.stop_reason)


class WallClockDeadlineExceeded(RuntimeError):
    """Raised when a unit must stop to persist before Modal timeout."""
