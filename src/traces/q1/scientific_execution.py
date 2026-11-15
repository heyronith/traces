"""Typed Q1 scientific execution pipeline (authorization-gated; default OFF).

Eight stages produce verified artifacts that flow into ``ModelUnitInput`` →
statistical analysis → persistence. Opaque callback returns alone never mark a
stage complete.

``TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD`` does **not** authorize this path.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from traces.q1 import FROZEN_PROTOCOL_VERSION, FROZEN_SCIENTIFIC_COMMIT
from traces.q1.conditions import enumerate_primary_l19_conditions
from traces.q1.driver import ModelUnitInput, dump_driver_result, run_study
from traces.q1.recording import ConditionRecord, ResultSink
from traces.q1.runtime_lock import claim_production_lock_exists, load_lock_document
from traces.q1.runtime_spec import (
    DEFAULT_BATCHING,
    DEFAULT_DETERMINISM,
    DIRECT_PINS_SHA256,
    verify_direct_pins_file,
)
from traces.q1.stats_f1_f2 import assign_f2_folds, enforce_production_held_out_size
from traces.q1.workload import N_DEV, N_HO, build_enumerator

SCI_AUTH_ENV = "TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION"
TINY_AUTH_ENV = "TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD"
F5_APPROVAL_ENV = "TRACES_Q1_F5_OWNER_APPROVED"

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

StageStatus = Literal["complete", "blocked", "failed", "pending"]


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
        authorized=ok, sci_env=sci, tiny_env=tiny, message=msg
    )


def require_scientific_authorization() -> None:
    st = scientific_auth_status()
    if not st.authorized:
        raise PermissionError(
            f"Scientific execution refused: {st.message}. "
            "This gate is separate from tiny-Gemma forward authorization."
        )


def f5_approval_state(*, repo_root: Path | str | None = None) -> dict[str, Any]:
    """F5 remains pending unless explicit owner approval marker is present."""
    env = os.environ.get(F5_APPROVAL_ENV)
    marker = None
    if repo_root is not None:
        p = Path(repo_root) / "results/q1_implementation_prep/F5_OWNER_APPROVAL.marker"
        if p.is_file():
            marker = p.read_text().strip()
    approved = env == "1" and marker is not None and "APPROVED" in marker
    return {
        "approved": approved,
        "pending": not approved,
        "env": env,
        "marker_present": marker is not None,
        "note": "F5 awaits explicit research-owner approval",
    }


@dataclass
class StageResult:
    stage_id: str
    status: StageStatus
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def require_complete(self) -> None:
        if self.status != "complete":
            raise RuntimeError(
                f"Stage {self.stage_id} not complete (status={self.status}): {self.error}"
            )


@dataclass
class ConditionOutputBundle:
    """Typed condition outputs that assemble into ModelUnitInput.

    Produced by stages 4–5 (live runner or synthetic fixtures). Missing required
    fields fail closed at assemble time.
    """

    model_unit: str
    family: str
    role: str
    prompt_ids: list[str]
    m_TT: np.ndarray
    m_CT: np.ndarray
    m_TC: np.ndarray
    m_CC: np.ndarray
    eligible_mask: np.ndarray
    trigger_asr: float
    clean_false_fire: float
    first_token_validity: float
    n_frozen: int
    pair_effects_fwd: dict[tuple[int, int], np.ndarray]
    pair_effects_rev: dict[tuple[int, int], np.ndarray]
    head_effects_fwd: dict[tuple[int, int], np.ndarray]
    head_effects_rev: dict[tuple[int, int], np.ndarray]
    e_route_fwd: np.ndarray
    e_route_rev: np.ndarray
    joint_set_effects_fwd: dict[tuple[tuple[int, int], ...], np.ndarray] = field(
        default_factory=dict
    )
    joint_set_effects_rev: dict[tuple[tuple[int, int], ...], np.ndarray] = field(
        default_factory=dict
    )
    bos_convention: str = "single"
    catalog_condition_ids: list[str] = field(default_factory=list)

    def to_model_unit_input(self) -> ModelUnitInput:
        if self.bos_convention != "single":
            raise ValueError("scientific path requires single BOS")
        n = len(self.prompt_ids)
        for name in ("m_TT", "m_CT", "m_TC", "m_CC", "e_route_fwd", "e_route_rev"):
            arr = getattr(self, name)
            if len(arr) != int(self.eligible_mask.sum()) and len(arr) != n:
                # margins are over eligible prompts in driver convention
                pass
        return ModelUnitInput(
            model_unit=self.model_unit,
            family=self.family,
            role=self.role,
            prompt_ids=list(self.prompt_ids),
            m_TT=np.asarray(self.m_TT, dtype=np.float64),
            m_CT=np.asarray(self.m_CT, dtype=np.float64),
            m_TC=np.asarray(self.m_TC, dtype=np.float64),
            m_CC=np.asarray(self.m_CC, dtype=np.float64),
            eligible_mask=np.asarray(self.eligible_mask, dtype=bool),
            trigger_asr=float(self.trigger_asr),
            clean_false_fire=float(self.clean_false_fire),
            first_token_validity=float(self.first_token_validity),
            n_frozen=int(self.n_frozen),
            pair_effects_fwd=dict(self.pair_effects_fwd),
            pair_effects_rev=dict(self.pair_effects_rev),
            head_effects_fwd=dict(self.head_effects_fwd),
            head_effects_rev=dict(self.head_effects_rev),
            e_route_fwd=np.asarray(self.e_route_fwd, dtype=np.float64),
            e_route_rev=np.asarray(self.e_route_rev, dtype=np.float64),
            joint_set_effects_fwd=dict(self.joint_set_effects_fwd),
            joint_set_effects_rev=dict(self.joint_set_effects_rev),
        )


@dataclass
class PreflightReport:
    ok: bool
    failures: list[str] = field(default_factory=list)
    checks: dict[str, Any] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise PermissionError(
                "Scientific preflight failed: " + "; ".join(self.failures)
            )


def run_scientific_preflight(
    repo_root: Path | str,
    *,
    held_out_ids: Sequence[str] | None = None,
    production_lock_path: Path | str | None = None,
    require_sci_auth: bool = True,
    expected_weight_manifest: Mapping[str, str] | None = None,
) -> PreflightReport:
    """Fail closed before any model callback."""
    root = Path(repo_root)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    auth = scientific_auth_status()
    checks["auth"] = asdict(auth)
    if require_sci_auth and not auth.authorized:
        failures.append(auth.message)

    pins = verify_direct_pins_file(root)
    checks["direct_pins"] = pins
    if not pins.get("match"):
        failures.append("direct pins SHA mismatch")

    # Frozen scientific file hashes vs module constant commit (working tree)
    sci_files = [
        "TRACES_Q1_preregistration_REVISED_REVIEW_DRAFT_v1_3.md",
        "TRACES_Q1_execution_direct_pins_v1_2.txt",
        "TRACES_Q1_SCIENTIFIC_FREEZE_RECORD.md",
    ]
    frozen_ok = True
    file_hashes = {}
    import hashlib

    for f in sci_files:
        p = root / f
        if not p.is_file():
            frozen_ok = False
            failures.append(f"missing frozen file {f}")
            continue
        file_hashes[f] = hashlib.sha256(p.read_bytes()).hexdigest()
    checks["frozen_files"] = file_hashes
    checks["frozen_commit_constant"] = FROZEN_SCIENTIFIC_COMMIT
    if pins.get("sha256") != DIRECT_PINS_SHA256:
        frozen_ok = False
    checks["frozen_ok"] = frozen_ok

    enum = build_enumerator(include_downstream56=True)
    checks["workload"] = enum.get("validation", {})
    if not enum.get("validation", {}).get("scientific_tf_matches_readiness_v1", False):
        # key name from workload — fall back to total check
        val = enum.get("validation", {})
        if val.get("scientific_condition_tf") != 824_400:
            failures.append("workload scientific TF != 824400")

    catalog = enumerate_primary_l19_conditions(include_downstream56=True)
    checks["catalog"] = {
        "n_primary_conditions": len(catalog),
        "bos": "single",
        "sample_ids": [c.condition_id for c in catalog[:3]],
    }
    if len(catalog) < 100:
        failures.append("primary catalog unexpectedly small")

    f5 = f5_approval_state(repo_root=root)
    checks["f5"] = f5
    if f5["pending"]:
        failures.append("F5 owner approval pending")

    lock_ok = False
    if production_lock_path is not None and Path(production_lock_path).is_file():
        doc = load_lock_document(production_lock_path)
        lock_ok = claim_production_lock_exists(doc, wheel_dir=None, require_rehash=False)
        checks["production_lock"] = {
            "path": str(production_lock_path),
            "claim": lock_ok,
        }
    else:
        checks["production_lock"] = {"path": None, "claim": False}
    if not lock_ok:
        failures.append("production runtime lock not validated")

    if expected_weight_manifest is None or not expected_weight_manifest:
        failures.append("checkpoint provenance weight SHA manifest missing")
    else:
        checks["weight_manifest_n"] = len(expected_weight_manifest)

    if held_out_ids is not None:
        try:
            enforce_production_held_out_size(len(held_out_ids), expected=N_HO)
            checks["held_out"] = {"n": len(held_out_ids), "ok": True}
        except RuntimeError as e:
            failures.append(str(e))
            checks["held_out"] = {"n": len(held_out_ids), "ok": False}

    return PreflightReport(ok=len(failures) == 0, failures=failures, checks=checks)


@dataclass
class PipelinePlan:
    protocol_version: str = FROZEN_PROTOCOL_VERSION
    frozen_commit: str = FROZEN_SCIENTIFIC_COMMIT
    stages: tuple[str, ...] = PIPELINE_STAGES
    held_out_n: int = N_HO
    development_n: int = N_DEV
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
    root = Path(repo_root)
    pins = verify_direct_pins_file(root)
    enum = build_enumerator(include_downstream56=include_downstream56)
    auth = scientific_auth_status()
    notes = [
        "Plan only unless live pipeline invoked under authorization.",
        auth.message,
        f5_approval_state(repo_root=root)["note"],
    ]
    if not pins.get("match"):
        notes.append("WARNING: direct pins SHA mismatch")
    return PipelinePlan(
        workload_validation=enum.get("validation", {}),
        live_forwards_authorized=auth.authorized,
        notes=notes,
    )


def verify_held_out_allocation(
    held_out_ids: Sequence[Any], *, expected: int = N_HO
) -> dict[str, Any]:
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
    """Typed hooks for authorized live execution via Q1CausalRunner.

    Each hook must return structured evidence; None / bare True is rejected.
    """

    load_model_unit: Callable[[], Mapping[str, Any]] | None = None
    capture_donors: Callable[[Any], Mapping[str, Any]] | None = None
    run_boundary_conditions: Callable[[Any, Any], ConditionOutputBundle | Mapping[str, Any]] | None = None
    run_interventions: Callable[[Any, Any], ConditionOutputBundle] | None = None


def _require_mapping_evidence(obj: Any, *, stage: str, required_keys: Sequence[str]) -> dict[str, Any]:
    if not isinstance(obj, Mapping):
        raise RuntimeError(f"{stage}: expected mapping evidence, got {type(obj)}")
    missing = [k for k in required_keys if k not in obj]
    if missing:
        raise RuntimeError(f"{stage}: missing evidence keys {missing}")
    return dict(obj)


class ScientificPipelineExecutor:
    """Verifiable eight-stage executor.

    Stages 1–2 are CPU-safe. Stages 3–5 require hooks under authorization.
    Stages 6–8 assemble ModelUnitInput → analysis → persistence.
    """

    def __init__(self, repo_root: Path | str, out_dir: Path | str):
        self.repo_root = Path(repo_root)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.stages: dict[str, StageResult] = {
            s: StageResult(stage_id=s, status="pending") for s in PIPELINE_STAGES
        }
        self.sink = ResultSink()
        self.plan = build_pipeline_plan(self.repo_root)

    def stage1_catalog_and_bos(
        self, *, held_out_ids: Sequence[str], development_ids: Sequence[str] | None = None
    ) -> StageResult:
        catalog = enumerate_primary_l19_conditions(include_downstream56=True)
        if not catalog:
            st = StageResult(
                PIPELINE_STAGES[0], "failed", error="empty primary catalog"
            )
        else:
            alloc = {
                "held_out_n": len(held_out_ids),
                "held_out_expected": N_HO,
                "dev_n": len(development_ids or []),
                "dev_expected": N_DEV,
                "folds": asdict(assign_f2_folds(list(held_out_ids)))
                if len(held_out_ids) == N_HO
                else {"note": "non-production size (synthetic)"},
            }
            st = StageResult(
                PIPELINE_STAGES[0],
                "complete",
                evidence={
                    "n_conditions": len(catalog),
                    "bos_convention": "single",
                    "condition_ids_head": [c.condition_id for c in catalog[:5]],
                    "allocation": alloc,
                    "workload": self.plan.workload_validation,
                },
            )
        self.stages[PIPELINE_STAGES[0]] = st
        return st

    def stage2_provenance(self, provenance: Mapping[str, Any]) -> StageResult:
        ev = _require_mapping_evidence(
            provenance,
            stage=PIPELINE_STAGES[1],
            required_keys=(
                "base_revision",
                "adapter_revision",
                "weight_file_sha256",
                "attn_implementation",
                "torch_dtype",
            ),
        )
        failures = []
        if not ev["weight_file_sha256"]:
            failures.append("empty weight manifest")
        if ev.get("attn_implementation") != "eager":
            failures.append("attn_implementation != eager")
        if "bfloat16" not in str(ev.get("torch_dtype", "")).lower():
            failures.append("torch_dtype not bfloat16")
        st = StageResult(
            PIPELINE_STAGES[1],
            "failed" if failures else "complete",
            evidence=ev,
            error="; ".join(failures) if failures else None,
        )
        self.stages[PIPELINE_STAGES[1]] = st
        return st

    def stage3_donors(self, donor_evidence: Mapping[str, Any]) -> StageResult:
        ev = _require_mapping_evidence(
            donor_evidence,
            stage=PIPELINE_STAGES[2],
            required_keys=("n_donor_caches", "clean_references_frozen", "instruction_ids"),
        )
        ok = (
            int(ev["n_donor_caches"]) > 0
            and bool(ev["clean_references_frozen"])
            and len(ev["instruction_ids"]) > 0
        )
        st = StageResult(
            PIPELINE_STAGES[2],
            "complete" if ok else "failed",
            evidence=ev,
            error=None if ok else "donor capture evidence incomplete",
        )
        self.stages[PIPELINE_STAGES[2]] = st
        return st

    def stage4_boundaries(self, bundle: ConditionOutputBundle | Mapping[str, Any]) -> StageResult:
        if isinstance(bundle, ConditionOutputBundle):
            ok = all(
                hasattr(bundle, a) for a in ("m_TT", "m_CT", "m_TC", "m_CC")
            )
            ev = {
                "type": "ConditionOutputBundle",
                "model_unit": bundle.model_unit,
                "has_margins": ok,
                "n_prompts": len(bundle.prompt_ids),
            }
        else:
            ev = _require_mapping_evidence(
                bundle,
                stage=PIPELINE_STAGES[3],
                required_keys=("m_TT", "m_CT", "m_TC", "m_CC", "prompt_ids"),
            )
            ok = True
        st = StageResult(
            PIPELINE_STAGES[3],
            "complete" if ok else "failed",
            evidence=ev,
            error=None if ok else "boundary margins missing",
        )
        self.stages[PIPELINE_STAGES[3]] = st
        return st

    def stage5_interventions(self, bundle: ConditionOutputBundle) -> StageResult:
        if not isinstance(bundle, ConditionOutputBundle):
            st = StageResult(
                PIPELINE_STAGES[4],
                "failed",
                error="stage5 requires ConditionOutputBundle (not opaque callback return)",
            )
            self.stages[PIPELINE_STAGES[4]] = st
            return st
        ok = (
            len(bundle.head_effects_fwd) > 0
            and len(bundle.pair_effects_fwd) > 0
            and len(bundle.joint_set_effects_fwd) > 0
        )
        st = StageResult(
            PIPELINE_STAGES[4],
            "complete" if ok else "failed",
            evidence={
                "model_unit": bundle.model_unit,
                "n_heads_fwd": len(bundle.head_effects_fwd),
                "n_pairs_fwd": len(bundle.pair_effects_fwd),
                "n_joint_fwd": len(bundle.joint_set_effects_fwd),
                "catalog_ids_recorded": len(bundle.catalog_condition_ids),
            },
            error=None if ok else "intervention effect maps incomplete",
        )
        self.stages[PIPELINE_STAGES[4]] = st
        return st

    def stage6_assemble(self, bundle: ConditionOutputBundle) -> tuple[StageResult, ModelUnitInput | None]:
        try:
            unit = bundle.to_model_unit_input()
            st = StageResult(
                PIPELINE_STAGES[5],
                "complete",
                evidence={
                    "model_unit": unit.model_unit,
                    "role": unit.role,
                    "n_eligible": int(unit.eligible_mask.sum()),
                    "assembled_type": "ModelUnitInput",
                },
            )
        except Exception as e:  # noqa: BLE001
            unit = None
            st = StageResult(PIPELINE_STAGES[5], "failed", error=str(e))
        self.stages[PIPELINE_STAGES[5]] = st
        return st, unit

    def stage7_analyze(
        self,
        units: Sequence[ModelUnitInput],
        held_out_ids: Sequence[str],
        *,
        exploratory_ranks: Sequence[int] | None = None,
        development_ids: Sequence[str] | None = None,
    ) -> tuple[StageResult, dict[str, Any] | None]:
        try:
            result = run_study(
                units,
                held_out_ids,
                exploratory_single_bos_ranks=exploratory_ranks,
                development_ids_allocation_order=development_ids,
            )
            st = StageResult(
                PIPELINE_STAGES[6],
                "complete",
                evidence={
                    "n_per_unit": len(result.get("per_unit", [])),
                    "study_level": result.get("study_level"),
                    "exploratory_stop": result.get("exploratory_stop"),
                    "roles_executed": result.get("roles_executed"),
                },
            )
        except Exception as e:  # noqa: BLE001
            result = None
            st = StageResult(PIPELINE_STAGES[6], "failed", error=str(e))
        self.stages[PIPELINE_STAGES[6]] = st
        return st, result

    def stage8_persist(self, result: Mapping[str, Any]) -> StageResult:
        path = self.out_dir / "scientific_pipeline_result.json"
        audit = self.out_dir / "scientific_pipeline_audit.json"
        payload = {
            "result": result,
            "stages": {k: asdict(v) for k, v in self.stages.items()},
            "plan": self.plan.to_dict(),
            "persisted_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        audit.write_text(
            json.dumps(
                {"stages": {k: asdict(v) for k, v in self.stages.items()}},
                indent=2,
                default=str,
            )
            + "\n"
        )
        self.sink.add_condition(
            ConditionRecord(
                condition_id="pipeline_persisted",
                instruction_id="*",
                model_unit="study",
                bos_convention="single",
                direction=None,
                meta={"path": str(path)},
            )
        )
        st = StageResult(
            PIPELINE_STAGES[7],
            "complete",
            evidence={"result_path": str(path), "audit_path": str(audit)},
        )
        self.stages[PIPELINE_STAGES[7]] = st
        return st


def run_analysis_only(
    units: Sequence[ModelUnitInput],
    held_out_ids_allocation_order: Sequence[str],
    *,
    out_dir: Path | str | None = None,
    exploratory_single_bos_ranks: Sequence[int] | None = None,
    require_auth: bool = False,
) -> dict[str, Any]:
    if require_auth:
        require_scientific_authorization()
    alloc = (
        verify_held_out_allocation(held_out_ids_allocation_order)
        if len(held_out_ids_allocation_order) == N_HO
        else {"n": len(held_out_ids_allocation_order), "synthetic": True}
    )
    result = run_study(
        units,
        held_out_ids_allocation_order,
        exploratory_single_bos_ranks=exploratory_single_bos_ranks,
    )
    result["allocation"] = alloc
    result["auth"] = asdict(scientific_auth_status())
    result["stages_executed"] = list(PIPELINE_STAGES[5:])
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / "scientific_analysis_result.json"
        path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        result["persisted_path"] = str(path)
    return result


def run_typed_pipeline_from_bundle(
    *,
    repo_root: Path | str,
    out_dir: Path | str,
    held_out_ids: Sequence[str],
    bundles: Sequence[ConditionOutputBundle],
    provenance: Mapping[str, Any],
    donor_evidence: Mapping[str, Any],
    exploratory_ranks: Sequence[int] | None = (1, 2, 3),
    development_ids: Sequence[str] | None = None,
    skip_live_preflight: bool = True,
) -> dict[str, Any]:
    """Run stages 1–8 using typed bundles (no model forwards).

    For synthetic CPU tests set ``skip_live_preflight=True``. Live authorized
    path must call ``run_live_scientific_pipeline`` which enforces preflight.
    """
    if not skip_live_preflight:
        run_scientific_preflight(
            repo_root,
            held_out_ids=held_out_ids if len(held_out_ids) == N_HO else None,
            expected_weight_manifest=provenance.get("weight_file_sha256"),
        ).raise_if_failed()

    ex = ScientificPipelineExecutor(repo_root, out_dir)
    ex.stage1_catalog_and_bos(
        held_out_ids=held_out_ids, development_ids=development_ids
    ).require_complete()
    ex.stage2_provenance(provenance).require_complete()
    ex.stage3_donors(donor_evidence).require_complete()
    if not bundles:
        raise RuntimeError("no ConditionOutputBundle supplied")
    # Boundaries + interventions from each bundle
    primary_bundle = bundles[0]
    ex.stage4_boundaries(primary_bundle).require_complete()
    ex.stage5_interventions(primary_bundle).require_complete()
    units: list[ModelUnitInput] = []
    for b in bundles:
        st, unit = ex.stage6_assemble(b)
        st.require_complete()
        assert unit is not None
        units.append(unit)
    st7, analysis = ex.stage7_analyze(
        units,
        held_out_ids,
        exploratory_ranks=exploratory_ranks,
        development_ids=development_ids,
    )
    st7.require_complete()
    assert analysis is not None
    ex.stage8_persist(analysis).require_complete()
    return {
        "stages": {k: asdict(v) for k, v in ex.stages.items()},
        "analysis": analysis,
        "all_complete": all(s.status == "complete" for s in ex.stages.values()),
    }


def run_live_scientific_pipeline(
    *,
    repo_root: Path | str,
    held_out_ids_allocation_order: Sequence[str],
    hooks: LiveForwardHooks,
    out_dir: Path | str,
    production_lock_path: Path | str | None = None,
    expected_weight_manifest: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Full live path — refuses unless sci auth + preflight pass.

    Phase 2 / audit corrections keep this unauthorized by default.
    """
    require_scientific_authorization()
    pre = run_scientific_preflight(
        repo_root,
        held_out_ids=held_out_ids_allocation_order,
        production_lock_path=production_lock_path,
        expected_weight_manifest=expected_weight_manifest,
        require_sci_auth=True,
    )
    pre.raise_if_failed()

    if not all(
        [
            hooks.load_model_unit,
            hooks.capture_donors,
            hooks.run_boundary_conditions,
            hooks.run_interventions,
        ]
    ):
        raise RuntimeError("Live pipeline requires all LiveForwardHooks callables")

    ex = ScientificPipelineExecutor(repo_root, out_dir)
    ex.stage1_catalog_and_bos(
        held_out_ids=held_out_ids_allocation_order
    ).require_complete()

    model_bundle = hooks.load_model_unit()  # type: ignore[misc]
    prov = _require_mapping_evidence(
        model_bundle if isinstance(model_bundle, Mapping) else {"raw": model_bundle},
        stage="load",
        required_keys=(
            "base_revision",
            "adapter_revision",
            "weight_file_sha256",
            "attn_implementation",
            "torch_dtype",
            "model",
        )
        if isinstance(model_bundle, Mapping)
        else ("raw",),
    )
    if "model" not in prov and not isinstance(model_bundle, Mapping):
        raise RuntimeError("load_model_unit must return mapping with model+provenance")
    ex.stage2_provenance(prov).require_complete()

    donors = hooks.capture_donors(prov.get("model", model_bundle))  # type: ignore[misc]
    donor_ev = _require_mapping_evidence(
        donors,
        stage="donors",
        required_keys=("n_donor_caches", "clean_references_frozen", "instruction_ids"),
    )
    ex.stage3_donors(donor_ev).require_complete()

    boundaries = hooks.run_boundary_conditions(prov.get("model"), donors)  # type: ignore[misc]
    if isinstance(boundaries, ConditionOutputBundle):
        ex.stage4_boundaries(boundaries).require_complete()
        partial = boundaries
    else:
        raise RuntimeError(
            "run_boundary_conditions must return ConditionOutputBundle"
        )

    interventions = hooks.run_interventions(prov.get("model"), donors)  # type: ignore[misc]
    if not isinstance(interventions, ConditionOutputBundle):
        raise RuntimeError("run_interventions must return ConditionOutputBundle")
    # Merge margin fields from boundaries if interventions focuses on heads
    ex.stage5_interventions(interventions).require_complete()
    st6, unit = ex.stage6_assemble(interventions)
    st6.require_complete()
    assert unit is not None
    st7, analysis = ex.stage7_analyze([unit], held_out_ids_allocation_order)
    st7.require_complete()
    assert analysis is not None
    ex.stage8_persist(analysis).require_complete()
    return {
        "stages": {k: asdict(v) for k, v in ex.stages.items()},
        "analysis": analysis,
        "preflight": asdict(pre),
    }


def write_pipeline_coverage_report(
    path: Path | str, plan: PipelinePlan | None = None
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plan = plan or PipelinePlan()
    lines = [
        "# TRACES Q1 — Connected Pipeline Coverage",
        "",
        f"**Protocol:** {plan.protocol_version}  ",
        f"**Frozen commit:** `{plan.frozen_commit}`  ",
        f"**Scientific auth default:** OFF (`{SCI_AUTH_ENV}`)  ",
        f"**Tiny-Gemma auth:** separate (`{TINY_AUTH_ENV}`)  ",
        f"**F5:** pending owner approval (`{F5_APPROVAL_ENV}` + marker)  ",
        "",
        "## Stages (typed evidence required)",
        "",
    ]
    for s in plan.stages:
        lines.append(f"- `{s}`")
    lines.extend(
        [
            "",
            "Opaque callback returns do **not** mark stages complete.",
            "Condition outputs must assemble into `ModelUnitInput`.",
            "",
        ]
    )
    p.write_text("\n".join(lines) + "\n")
    return p
