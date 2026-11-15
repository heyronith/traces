"""End-to-end Q1 scientific execution orchestration (authorization-gated).

Connects frozen catalog → provenance → donors → TT/CT/TC/CC → interventions →
effects → F1/F2 decisions → persistence.

**Default: OFF.** Setting ``TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD`` does **not**
authorize this path. Scientific execution requires the separate env var
``TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION=1``.

Implementing this module does **not** authorize invoking live model forwards.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from traces.q1 import FROZEN_PROTOCOL_VERSION, FROZEN_SCIENTIFIC_COMMIT
from traces.q1.driver import ModelUnitInput, run_study
from traces.q1.runtime_spec import (
    DEFAULT_BATCHING,
    DEFAULT_DETERMINISM,
    verify_direct_pins_file,
)
from traces.q1.stats_f1_f2 import assign_f2_folds, enforce_production_held_out_size
from traces.q1.workload import build_enumerator

SCI_AUTH_ENV = "TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION"
TINY_AUTH_ENV = "TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD"

PIPELINE_STAGES = (
    "1_frozen_prompt_catalog_and_bos_verification",
    "2_model_unit_configuration_and_provenance",
    "3_donor_capture_and_clean_reference_freezing",
    "4_tt_ct_tc_cc_conditions",
    "5_single_head_pair_joint_residual_interventions",
    "6_frozen_effect_calculations",
    "7_f1_f2_uncertainty_and_decision_rules",
    "8_structured_result_persistence_and_audit_logging",
)


@dataclass(frozen=True)
class ScientificAuthStatus:
    authorized: bool
    sci_env: str | None
    tiny_env: str | None
    tiny_does_not_authorize_scientific: bool = True
    message: str = ""


def scientific_auth_status() -> ScientificAuthStatus:
    sci = os.environ.get(SCI_AUTH_ENV)
    tiny = os.environ.get(TINY_AUTH_ENV)
    ok = sci == "1"
    if ok:
        msg = "scientific execution authorized"
    elif tiny == "1":
        msg = (
            f"{TINY_AUTH_ENV}=1 does not authorize scientific execution; "
            f"require {SCI_AUTH_ENV}=1"
        )
    else:
        msg = f"{SCI_AUTH_ENV} unset/off (default OFF)"
    return ScientificAuthStatus(
        authorized=ok,
        sci_env=sci,
        tiny_env=tiny,
        message=msg,
    )


def require_scientific_authorization() -> None:
    st = scientific_auth_status()
    if not st.authorized:
        raise PermissionError(
            f"Scientific execution refused: {st.message}. "
            "This gate is separate from tiny-Gemma forward authorization."
        )


@dataclass
class PipelinePlan:
    """Complete execution plan preserving frozen workload and dataset allocation."""

    protocol_version: str = FROZEN_PROTOCOL_VERSION
    frozen_commit: str = FROZEN_SCIENTIFIC_COMMIT
    stages: tuple[str, ...] = PIPELINE_STAGES
    held_out_n: int = 200
    bos_convention: str = "single"
    batching_policy_id: str = field(default_factory=lambda: DEFAULT_BATCHING.policy_id())
    determinism: dict[str, Any] = field(
        default_factory=lambda: asdict(DEFAULT_DETERMINISM)
    )
    workload_validation: dict[str, Any] = field(default_factory=dict)
    live_forwards_authorized: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pipeline_plan(
    repo_root: Path | str,
    *,
    include_downstream56: bool = True,
) -> PipelinePlan:
    """Build the connected plan without executing model forwards."""
    root = Path(repo_root)
    pins = verify_direct_pins_file(root)
    enum = build_enumerator(include_downstream56=include_downstream56)
    auth = scientific_auth_status()
    notes = [
        "Plan only — does not invoke Q1CausalRunner or load checkpoints.",
        auth.message,
    ]
    if not pins.get("match"):
        notes.append("WARNING: direct pins file SHA mismatch vs frozen pin")
    return PipelinePlan(
        workload_validation=enum.get("validation", {}),
        live_forwards_authorized=auth.authorized,
        notes=notes,
    )


def verify_held_out_allocation(held_out_ids: Sequence[Any], *, expected: int = 200) -> dict[str, Any]:
    """Preserve frozen allocation order; enforce production size."""
    enforce_production_held_out_size(len(held_out_ids), expected=expected)
    folds = assign_f2_folds(list(held_out_ids))
    return {
        "n": len(held_out_ids),
        "fold_a_n": len(folds.fold_a_ids),
        "fold_b_n": len(folds.fold_b_ids),
        "allocation_order_preserved": True,
    }


@dataclass
class LiveForwardHooks:
    """Optional callables for authorized live execution (not invoked by default).

    Each hook must be supplied by the caller under scientific authorization.
    Synthetic / precomputed analysis uses ``run_analysis_only`` instead.
    """

    load_model_unit: Callable[..., Any] | None = None
    capture_donors: Callable[..., Any] | None = None
    run_boundary_conditions: Callable[..., Any] | None = None
    run_interventions: Callable[..., Any] | None = None


def run_analysis_only(
    units: Sequence[ModelUnitInput],
    held_out_ids_allocation_order: Sequence[str],
    *,
    out_dir: Path | str | None = None,
    exploratory_single_bos_ranks: Sequence[int] | None = None,
    require_auth: bool = False,
) -> dict[str, Any]:
    """Run stages 6–8 on precomputed effects (no model forwards).

    When ``require_auth=True``, refuses unless scientific auth is ON.
    Default ``require_auth=False`` allows synthetic CPU analysis tests.
    """
    if require_auth:
        require_scientific_authorization()
    alloc = verify_held_out_allocation(held_out_ids_allocation_order)
    result = run_study(
        units,
        held_out_ids_allocation_order,
        exploratory_single_bos_ranks=exploratory_single_bos_ranks,
    )
    result["allocation"] = alloc
    result["auth"] = asdict(scientific_auth_status())
    result["stages_executed"] = [
        "6_frozen_effect_calculations",
        "7_f1_f2_uncertainty_and_decision_rules",
        "8_structured_result_persistence_and_audit_logging",
    ]
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "scientific_analysis_result.json"
        path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        result["persisted_path"] = str(path)
    return result


def run_live_scientific_pipeline(
    *,
    repo_root: Path | str,
    held_out_ids_allocation_order: Sequence[str],
    hooks: LiveForwardHooks,
    out_dir: Path | str,
) -> dict[str, Any]:
    """Full live path — **refuses** unless ``TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION=1``.

    Even when authorized, requires all hooks to be non-None. This function is
    the explicit gate; Phase 2 engineering does not call it.
    """
    require_scientific_authorization()
    if not all(
        [
            hooks.load_model_unit,
            hooks.capture_donors,
            hooks.run_boundary_conditions,
            hooks.run_interventions,
        ]
    ):
        raise RuntimeError(
            "Live scientific pipeline requires all LiveForwardHooks callables"
        )

    plan = build_pipeline_plan(repo_root)
    alloc = verify_held_out_allocation(held_out_ids_allocation_order)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    audit = {
        "artifact": "TRACES_Q1_live_scientific_pipeline",
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "plan": plan.to_dict(),
        "allocation": alloc,
        "stages": {},
    }
    # Stage markers only — actual hook bodies are caller-supplied under auth
    audit["stages"]["1"] = {"status": "caller_must_verify_bos_catalog"}
    model_bundle = hooks.load_model_unit()  # type: ignore[misc]
    audit["stages"]["2"] = {"status": "loaded", "type": type(model_bundle).__name__}
    donors = hooks.capture_donors(model_bundle)  # type: ignore[misc]
    audit["stages"]["3"] = {"status": "donors_captured"}
    boundaries = hooks.run_boundary_conditions(model_bundle, donors)  # type: ignore[misc]
    audit["stages"]["4"] = {"status": "boundaries_done"}
    interventions = hooks.run_interventions(model_bundle, donors)  # type: ignore[misc]
    audit["stages"]["5"] = {"status": "interventions_done"}
    audit["stages"]["6_8"] = {
        "status": "effects_must_be_fed_to_run_analysis_only",
        "note": "Convert intervention outputs to ModelUnitInput externally",
    }
    path = out / "live_pipeline_audit.json"
    path.write_text(json.dumps(audit, indent=2, default=str) + "\n")
    return {"audit_path": str(path), "audit": audit, "boundaries": boundaries, "interventions": interventions}


def write_pipeline_coverage_report(path: Path | str, plan: PipelinePlan | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plan = plan or PipelinePlan()
    lines = [
        "# TRACES Q1 — Connected Pipeline Coverage",
        "",
        f"**Protocol:** {plan.protocol_version}  ",
        f"**Frozen commit:** `{plan.frozen_commit}`  ",
        f"**Scientific auth default:** OFF (`{SCI_AUTH_ENV}`)  ",
        f"**Tiny-Gemma auth:** separate (`{TINY_AUTH_ENV}`) — does not unlock this path  ",
        "",
        "## Stages",
        "",
    ]
    for s in plan.stages:
        lines.append(f"- `{s}`")
    lines.extend(
        [
            "",
            "## Workload (from enumerator validation when built)",
            "",
            "```json",
            json.dumps(plan.workload_validation, indent=2),
            "```",
            "",
            "## Authorization",
            "",
            "- Live forwards: **not invoked** by Phase 2 engineering.",
            "- Analysis-only path available for synthetic / precomputed effects.",
            "",
        ]
    )
    p.write_text("\n".join(lines) + "\n")
    return p
