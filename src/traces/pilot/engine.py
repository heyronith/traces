"""Stage 1 pilot evaluation engine facade (disabled by default).

Real GPU execution uses ``run_real_pilot_pipeline`` → two-phase discovery/transfer.
Synthetic CPU tests use ``run_pilot_pipeline(..., execution_mode='synthetic_test')``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from traces.pilot.components import ComponentRef
from traces.pilot.runner import load_pilot_config
from traces.pilot.schedule import enumerate_engine_condition_schedule
from traces.pilot.transfer import (
    DEFAULT_K_BUDGETS,
    SEEDS,
    correlated_pair_note,
)

ExecutionMode = Literal["synthetic_test", "real"]

__all__ = [
    "PilotScorer",
    "PilotEngineResult",
    "enumerate_engine_condition_schedule",
    "run_pilot_pipeline",
    "run_real_pilot_pipeline",
    "_scorer_provenance",
]


class PilotScorer(Protocol):
    """Scoring backend: real model or deterministic synthetic stub."""

    def score_D_single(
        self, seed: int, component_id: str, instruction_id: str, split: str
    ) -> float: ...

    def score_D_set(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> float: ...

    def score_K_set(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> float: ...

    def score_asr(
        self, seed: int, component_ids: Sequence[str], instruction_ids: Sequence[str], split: str
    ) -> dict[str, float]: ...


@dataclass
class PilotEngineResult:
    per_seed: dict[str, Any] = field(default_factory=dict)
    per_pair: dict[str, Any] = field(default_factory=dict)
    per_condition: list[dict[str, Any]] = field(default_factory=list)
    reliability: dict[str, Any] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    analysis: dict[str, Any] = field(default_factory=dict)
    metric_definitions: dict[str, Any] = field(default_factory=dict)
    schedule_counts: dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    executed: bool = False
    stopped_early: bool = False
    stop_reason: str | None = None
    execution_mode: str = "synthetic_test"

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_seed": self.per_seed,
            "per_pair": self.per_pair,
            "per_condition": self.per_condition,
            "reliability": self.reliability,
            "gates": self.gates,
            "analysis": self.analysis,
            "metric_definitions": self.metric_definitions,
            "schedule_counts": self.schedule_counts,
            "runtime_seconds": self.runtime_seconds,
            "executed": self.executed,
            "stopped_early": self.stopped_early,
            "stop_reason": self.stop_reason,
            "execution_mode": self.execution_mode,
            "pair_dependence_note": correlated_pair_note(),
        }


def _scorer_provenance(scorer: Any, seed: int) -> dict[str, str]:
    if hasattr(scorer, "provenance_for_seed"):
        try:
            p = scorer.provenance_for_seed(seed)
            return {
                "base_model_revision": str(p.get("base_model_revision") or "unspecified"),
                "adapter_subfolder": str(p.get("adapter_subfolder") or "unspecified"),
                "adapter_revision": str(p.get("adapter_revision") or "unspecified"),
            }
        except Exception:
            pass
    return {
        "base_model_revision": "synthetic",
        "adapter_subfolder": f"l1523/seed{seed}",
        "adapter_revision": "synthetic",
    }


def assert_synthetic_execution_allowed(scorer: Any) -> None:
    cls_name = type(scorer).__name__
    if cls_name == "RealGemmaPilotScorer":
        raise RuntimeError(
            "Refusing synthetic_test execution mode with RealGemmaPilotScorer. "
            "Use run_real_pilot_pipeline (auth-gated)."
        )


def run_real_pilot_pipeline(
    scorer: PilotScorer,
    **kwargs: Any,
) -> PilotEngineResult:
    """Separately authorized real-pilot entrypoint (never unlocked by calibration)."""
    from traces.pilot.execution import run_real_pilot_authorized

    return run_real_pilot_authorized(scorer, **kwargs)


def run_pilot_pipeline(
    scorer: PilotScorer,
    *,
    seeds: Sequence[int] = SEEDS,
    k_budgets: Sequence[int] = DEFAULT_K_BUDGETS,
    catalog: Sequence[ComponentRef] | None = None,
    output_dir: Path | str | None = None,
    config_path: Path | str | None = None,
    execution_mode: ExecutionMode = "real",
    allow_execution: bool | None = None,
) -> PilotEngineResult:
    """Facade: real → auth-gated two-phase; synthetic_test → two-phase CPU stubs."""
    load_pilot_config(config_path)

    if allow_execution is True and execution_mode == "real":
        raise RuntimeError(
            "allow_execution=True cannot unlock real-model pilot execution. "
            "Use an authorized configs/pilot.yaml via the real entrypoint after review."
        )

    if execution_mode == "real":
        return run_real_pilot_pipeline(
            scorer,
            seeds=seeds,
            k_budgets=k_budgets,
            catalog=catalog,
            output_dir=output_dir,
            config_path=config_path,
        )

    assert_synthetic_execution_allowed(scorer)
    if allow_execution is False:
        raise RuntimeError("Pilot synthetic test execution explicitly disallowed")

    from traces.pilot.execution import run_two_phase_pilot

    return run_two_phase_pilot(
        scorer,
        seeds=seeds,
        k_budgets=k_budgets,
        catalog=catalog,
        output_dir=output_dir,
        resume=True,
        execution_mode="synthetic_test",
    )
