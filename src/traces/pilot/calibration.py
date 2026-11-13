"""Bounded real-model calibration workload (implemented, not authorized to run).

Uses one dense-LoRA seed, a few ordinary instructions, fixed attention/MLP/joint
sets, actual teacher-forced D, benign K, and greedy generation. Hard timeout and
conservative cost estimate included. Modal CLI must refuse launch until authorized.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from traces.cost_estimator import estimate_cost, usd_per_hour_for_gpu
from traces.pilot.components import ComponentRef, build_full_l1523_catalog
from traces.pilot.controls import DAMAGE_MATCH_N_CANDIDATES

# Inline constants so refuse_calibration_launch works without importing torch
# (metrics_defs pulls torch; Modal local entrypoint may lack it).
SCORING_DEFINITION_ID = "traces_pilot_D_K_v2"
INTERVENTION_SEMANTICS = "activation_ablation_joint_hooks_not_weight_reversion"

CALIBRATION_SEED = 42
CALIBRATION_N_INSTRUCTIONS = 4
CALIBRATION_TIMEOUT_SECONDS = 12 * 60
CALIBRATION_AUTHORIZED_TOKEN = "STAGE1_CALIBRATION_AUTHORIZED_BY_INDEPENDENT_REVIEW"


@dataclass
class CalibrationPlan:
    seed: int = CALIBRATION_SEED
    adapter_subfolder: str = "l1523/seed42"
    n_instructions: int = CALIBRATION_N_INSTRUCTIONS
    timeout_seconds: int = CALIBRATION_TIMEOUT_SECONDS
    scoring_definition_id: str = SCORING_DEFINITION_ID
    intervention_semantics: str = INTERVENTION_SEMANTICS
    attention_set: tuple[str, ...] = ()
    mlp_set: tuple[str, ...] = ()
    joint_set: tuple[str, ...] = ()
    measures: tuple[str, ...] = ("teacher_forced_D", "benign_K", "greedy_generation")
    status: str = "prepared_disabled"
    notes: str = (
        "Bounded calibration for cost/hook timing only. Not the scientific pilot. "
        "Must not run without separate calibration authorization."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_calibration_plan(catalog: Sequence[ComponentRef] | None = None) -> CalibrationPlan:
    cat = list(catalog or build_full_l1523_catalog())
    attn = tuple(c.component_id for c in cat if c.kind == "attention_head")[:2]
    mlp = tuple(c.component_id for c in cat if c.kind == "mlp_channel_group")[:2]
    joint = attn[:1] + mlp[:1]
    return CalibrationPlan(
        attention_set=attn,
        mlp_set=mlp,
        joint_set=joint,
    )


def estimate_calibration_cost(*, gpu: str = "A10G") -> dict[str, Any]:
    """Conservative software cost estimate (not an account spend cap)."""
    # Rough: load + ~40 TF forwards + ~16 greedy gens
    assumed_seconds = 180.0 + 40 * 0.056 + 16 * 1.633 + 60.0  # + contingency
    rate = usd_per_hour_for_gpu(gpu)
    est = estimate_cost(assumed_seconds, budget_usd=1e9, gpu=gpu, usd_per_hour=rate)
    return {
        "gpu": gpu,
        "assumed_seconds": assumed_seconds,
        "usd_projected": est.projected_usd,
        "usd_per_hour": rate,
        "timeout_seconds": CALIBRATION_TIMEOUT_SECONDS,
        "note": "Software estimate only; not an enforced Modal account limit.",
    }


def assert_calibration_authorized(cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or {}
    cal = cfg.get("calibration") or {}
    if cal.get("enabled") is not True:
        raise RuntimeError("Calibration disabled (calibration.enabled != true)")
    if cal.get("authorization_token") != CALIBRATION_AUTHORIZED_TOKEN:
        raise RuntimeError("Calibration authorization_token missing/invalid")


def run_bounded_calibration(
    scorer: Any,
    *,
    plan: CalibrationPlan | None = None,
    instruction_ids: Sequence[str] | None = None,
    split: str = "discovery",
    authorized: bool = False,
) -> dict[str, Any]:
    """Execute the tiny calibration on an already-constructed scorer.

    Callers (Modal) must pass authorized=True only after assert_calibration_authorized.
    This function still refuses if authorized is False — no hidden launch path.
    """
    if not authorized:
        raise RuntimeError(
            "run_bounded_calibration refused: authorized=False. "
            "No GPU calibration without independent authorization."
        )
    plan = plan or build_calibration_plan()
    t0 = time.perf_counter()
    if time.perf_counter() - t0 > plan.timeout_seconds:
        raise TimeoutError("Calibration timeout before start")

    # Prefer first N discovery IDs if not provided
    if instruction_ids is None:
        from traces.pilot.dataset import get_split_instructions

        instruction_ids = [
            i.instruction_id for i in get_split_instructions(split)  # type: ignore[arg-type]
        ][: plan.n_instructions]
    ids = list(instruction_ids)[: plan.n_instructions]
    if len(ids) < 2:
        raise RuntimeError("Calibration requires at least 2 instructions")

    results: dict[str, Any] = {
        "plan": plan.to_dict(),
        "instruction_ids": ids,
        "cost_estimate": estimate_calibration_cost(),
        "sets": {},
        "authorized": True,
    }

    def _score_set(name: str, comps: Sequence[str]) -> dict[str, Any]:
        if time.perf_counter() - t0 > plan.timeout_seconds:
            raise TimeoutError(f"Calibration hard timeout during {name}")
        D = scorer.score_D_set(plan.seed, list(comps), ids, split)
        K = scorer.score_K_set(plan.seed, list(comps), ids, split)
        asr = scorer.score_asr(plan.seed, list(comps), ids, split)
        return {"D": D, "K": K, **asr, "components": list(comps)}

    results["sets"]["attention"] = _score_set("attention", plan.attention_set)
    results["sets"]["mlp"] = _score_set("mlp", plan.mlp_set)
    results["sets"]["joint"] = _score_set("joint", plan.joint_set)
    results["wall_seconds"] = time.perf_counter() - t0
    results["timeout_seconds"] = plan.timeout_seconds
    results["n_damage_candidates_not_run"] = DAMAGE_MATCH_N_CANDIDATES
    results["status"] = "completed_bounded_calibration"
    return results


def refuse_calibration_launch() -> dict[str, Any]:
    plan = build_calibration_plan()
    return {
        "mode": "pilot_calibration",
        "status": "REFUSED_NOT_AUTHORIZED",
        "plan": plan.to_dict(),
        "cost_estimate": estimate_calibration_cost(),
        "note": (
            "Calibration implementation is present but must not run until separately "
            "authorized. No GPU launched."
        ),
        "intended_later_command": "modal run modal_app.py --pilot-calibration",
    }
