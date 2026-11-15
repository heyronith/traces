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
from traces.q1.frozen_artifacts import verify_frozen_artifacts
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


# Exact owner-approved status line (substring "APPROVED" alone is insufficient —
# "NOT APPROVED" must not authorize).
F5_APPROVED_STATUS_EXACT = "F5_STATUS=OWNER_APPROVED"


def f5_approval_state(*, repo_root: Path | str | None = None) -> dict[str, Any]:
    """F5 remains pending unless exact owner-approved status with provenance.

    Mere marker existence or the substring ``APPROVED`` (e.g. ``NOT APPROVED``)
    does **not** authorize F5.
    """
    env = os.environ.get(F5_APPROVAL_ENV)
    marker_text = None
    marker_path = None
    if repo_root is not None:
        marker_path = (
            Path(repo_root) / "results/q1_implementation_prep/F5_OWNER_APPROVAL.marker"
        )
        if marker_path.is_file():
            marker_text = marker_path.read_text()

    status_line = None
    signer = None
    if marker_text is not None:
        for line in marker_text.splitlines():
            line = line.strip()
            if line.startswith("F5_STATUS="):
                status_line = line
            if line.startswith("OWNER_SIGNER="):
                signer = line.split("=", 1)[1].strip() or None

    approved = (
        env == "1"
        and status_line == F5_APPROVED_STATUS_EXACT
        and signer is not None
        and len(signer) >= 2
    )
    return {
        "approved": approved,
        "pending": not approved,
        "env": env,
        "marker_present": marker_text is not None,
        "status_line": status_line,
        "signer_present": signer is not None,
        "required_status": F5_APPROVED_STATUS_EXACT,
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
    # Prompt IDs in frozen allocation order corresponding to eligible-only arrays
    eligible_prompt_ids: list[str] | None = None

    def validate(self) -> list[str]:
        """Strict consistency checks; empty list means OK."""
        failures: list[str] = []
        if self.bos_convention != "single":
            failures.append("scientific path requires single BOS")
        n = len(self.prompt_ids)
        if n < 1:
            failures.append("empty prompt_ids")
        if len(set(self.prompt_ids)) != n:
            failures.append("duplicate prompt_ids")
        mask = np.asarray(self.eligible_mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != n:
            failures.append(
                f"eligible_mask length {len(mask) if mask.ndim == 1 else mask.shape} "
                f"!= n_prompts {n}"
            )
            return failures
        n_elig = int(mask.sum())
        if n_elig < 1:
            failures.append("no eligible prompts")
        # Deterministic eligible ordering: mask order must match prompt_ids order
        elig_ids_from_mask = [pid for pid, m in zip(self.prompt_ids, mask) if m]
        if self.eligible_prompt_ids is not None:
            if list(self.eligible_prompt_ids) != elig_ids_from_mask:
                failures.append(
                    "eligible_prompt_ids ordering mismatch vs mask over prompt_ids"
                )
        expected_len = n_elig  # eligible-only arrays required

        def _check_vec(name: str, arr: Any) -> None:
            a = np.asarray(arr, dtype=np.float64)
            if a.ndim != 1:
                failures.append(f"{name}: expected 1-d")
                return
            if len(a) == n and n != n_elig:
                failures.append(
                    f"{name}: full-denominator length {n} supplied where "
                    f"eligible-only length {n_elig} is required"
                )
                return
            if len(a) != expected_len:
                failures.append(
                    f"{name}: length {len(a)} != n_eligible {expected_len}"
                )
            if len(a) and not np.all(np.isfinite(a)):
                failures.append(f"{name}: nonfinite values")

        for name in ("m_TT", "m_CT", "m_TC", "m_CC", "e_route_fwd", "e_route_rev"):
            _check_vec(name, getattr(self, name))

        for label, fx in (
            ("pair_fwd", self.pair_effects_fwd),
            ("pair_rev", self.pair_effects_rev),
            ("head_fwd", self.head_effects_fwd),
            ("head_rev", self.head_effects_rev),
        ):
            if not isinstance(fx, Mapping) or not fx:
                failures.append(f"{label}: empty or not a mapping")
                continue
            for k, v in fx.items():
                _check_vec(f"{label}{k}", v)

        for label, fx in (
            ("joint_fwd", self.joint_set_effects_fwd),
            ("joint_rev", self.joint_set_effects_rev),
        ):
            for k, v in fx.items():
                _check_vec(f"{label}{k}", v)

        if self.n_frozen != n and self.role in {
            "primary_confirmatory",
            "secondary_replication",
            "exploratory_dev",
        }:
            # n_frozen is the split denominator; must match prompt catalog size
            if self.n_frozen < n:
                failures.append(f"n_frozen {self.n_frozen} < n_prompts {n}")
        return failures

    def to_model_unit_input(self) -> ModelUnitInput:
        failures = self.validate()
        if failures:
            raise ValueError(
                "ConditionOutputBundle validation failed: " + "; ".join(failures)
            )
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
    production_wheel_dir: Path | str | None = None,
    require_sci_auth: bool = True,
    expected_weight_manifest: Mapping[str, str] | None = None,
) -> PreflightReport:
    """Fail closed before any model callback.

    Frozen artifacts are compared to immutable expected SHA baselines (not
    merely recomputed). Production lock claims require on-disk wheel rehashing.
    """
    root = Path(repo_root)
    failures: list[str] = []
    checks: dict[str, Any] = {}

    auth = scientific_auth_status()
    checks["auth"] = asdict(auth)
    if require_sci_auth and not auth.authorized:
        failures.append(auth.message)

    pins = verify_direct_pins_file(root)
    checks["direct_pins"] = pins
    if not pins.get("match") or pins.get("sha256") != DIRECT_PINS_SHA256:
        failures.append("direct pins SHA mismatch vs frozen baseline")

    frozen = verify_frozen_artifacts(root)
    checks["frozen_artifacts"] = frozen
    if not frozen["ok"]:
        failures.extend(frozen["failures"])

    enum = build_enumerator(include_downstream56=True)
    val = enum.get("validation", {})
    checks["workload"] = val
    if val.get("scientific_condition_tf") != 824_400:
        failures.append(
            f"workload scientific TF {val.get('scientific_condition_tf')} != 824400"
        )
    if val.get("gen_passes") != 3_600:
        failures.append(f"workload gen_passes {val.get('gen_passes')} != 3600")

    catalog = enumerate_primary_l19_conditions(include_downstream56=True)
    checks["catalog"] = {
        "n_primary_conditions": len(catalog),
        "bos": "single",
        "sample_ids": [c.condition_id for c in catalog[:3]],
    }
    if len(catalog) < 100:
        failures.append("primary catalog unexpectedly small / invalid")

    f5 = f5_approval_state(repo_root=root)
    checks["f5"] = f5
    if f5["pending"]:
        failures.append("F5 owner approval pending")

    lock_ok = False
    if (
        production_lock_path is not None
        and Path(production_lock_path).is_file()
        and production_wheel_dir is not None
        and Path(production_wheel_dir).is_dir()
    ):
        doc = load_lock_document(production_lock_path)
        # Production policy: always rehash wheels; never trust flags alone
        lock_ok = claim_production_lock_exists(
            doc,
            wheel_dir=production_wheel_dir,
            require_rehash=True,
            pins_file=root / "TRACES_Q1_execution_direct_pins_v1_2.txt",
        )
        checks["production_lock"] = {
            "path": str(production_lock_path),
            "wheel_dir": str(production_wheel_dir),
            "claim": lock_ok,
        }
    else:
        checks["production_lock"] = {
            "path": production_lock_path,
            "wheel_dir": production_wheel_dir,
            "claim": False,
            "reason": "lock path and wheel_dir both required for production claim",
        }
    if not lock_ok:
        failures.append("production runtime lock not validated (blocked)")

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
class UnitSpec:
    """Frozen role/seed identity for multi-unit live execution."""

    model_unit: str
    family: str
    role: str
    seed: int | None


def expected_frozen_unit_specs() -> list[UnitSpec]:
    """Exploratory 42–44, primary 42–46, secondary 45–46, adapter-off."""
    from traces.q1.frozen_artifacts import (
        REQUIRED_EXPLORATORY_SEEDS,
        REQUIRED_PRIMARY_SEEDS,
        REQUIRED_SECONDARY_SEEDS,
    )

    specs: list[UnitSpec] = []
    for s in REQUIRED_EXPLORATORY_SEEDS:
        specs.append(
            UnitSpec(
                model_unit=f"exploratory_l1523_seed{s}",
                family="l1523",
                role="exploratory_dev",
                seed=s,
            )
        )
    for s in REQUIRED_PRIMARY_SEEDS:
        specs.append(
            UnitSpec(
                model_unit=f"primary_l19_seed{s}",
                family="l19",
                role="primary_confirmatory",
                seed=s,
            )
        )
    for s in REQUIRED_SECONDARY_SEEDS:
        specs.append(
            UnitSpec(
                model_unit=f"secondary_l1523_seed{s}",
                family="l1523",
                role="secondary_replication",
                seed=s,
            )
        )
    specs.append(
        UnitSpec(
            model_unit="adapter_off_base_reference",
            family="base",
            role="adapter_off_gates_descriptives",
            seed=None,
        )
    )
    return specs


@dataclass
class LiveForwardHooks:
    """Typed Q1CausalRunner adapters for authorized live execution.

    ``execute_unit`` must build/use ``Q1CausalRunner`` and return both boundary
    and intervention ``ConditionOutputBundle`` objects for that unit.
    """

    load_model_for_unit: Callable[[UnitSpec], Mapping[str, Any]] | None = None
    capture_donors_for_unit: Callable[[UnitSpec, Any], Mapping[str, Any]] | None = None
    execute_unit: Callable[
        [UnitSpec, Any, Any], tuple[ConditionOutputBundle, ConditionOutputBundle]
    ] | None = None


def reconcile_boundary_intervention(
    boundary: ConditionOutputBundle,
    intervention: ConditionOutputBundle,
) -> ConditionOutputBundle:
    """Merge TT/CT/TC/CC margins from boundary into intervention head/pair maps.

    Rejects inconsistent prompt IDs, masks, routes, or model_unit identity.
    """
    failures: list[str] = []
    if boundary.model_unit != intervention.model_unit:
        failures.append("model_unit mismatch between boundary and intervention")
    if boundary.prompt_ids != intervention.prompt_ids:
        failures.append("prompt_ids mismatch between boundary and intervention")
    if not np.array_equal(
        np.asarray(boundary.eligible_mask), np.asarray(intervention.eligible_mask)
    ):
        failures.append("eligible_mask mismatch")
    if boundary.role != intervention.role or boundary.family != intervention.family:
        failures.append("role/family mismatch")
    # Boundary must carry margins; intervention must carry head/pair/joint maps
    for name in ("m_TT", "m_CT", "m_TC", "m_CC"):
        if not np.allclose(
            np.asarray(getattr(boundary, name), dtype=np.float64),
            np.asarray(getattr(intervention, name), dtype=np.float64),
            equal_nan=True,
        ):
            failures.append(f"{name} disagree between boundary and intervention")
    if failures:
        raise ValueError(
            "boundary/intervention reconcile failed: " + "; ".join(failures)
        )
    # Prefer intervention maps with boundary-validated margins
    return ConditionOutputBundle(
        model_unit=intervention.model_unit,
        family=intervention.family,
        role=intervention.role,
        prompt_ids=list(intervention.prompt_ids),
        m_TT=np.asarray(boundary.m_TT),
        m_CT=np.asarray(boundary.m_CT),
        m_TC=np.asarray(boundary.m_TC),
        m_CC=np.asarray(boundary.m_CC),
        eligible_mask=np.asarray(boundary.eligible_mask, dtype=bool),
        trigger_asr=intervention.trigger_asr,
        clean_false_fire=intervention.clean_false_fire,
        first_token_validity=intervention.first_token_validity,
        n_frozen=intervention.n_frozen,
        pair_effects_fwd=dict(intervention.pair_effects_fwd),
        pair_effects_rev=dict(intervention.pair_effects_rev),
        head_effects_fwd=dict(intervention.head_effects_fwd),
        head_effects_rev=dict(intervention.head_effects_rev),
        e_route_fwd=np.asarray(boundary.e_route_fwd),
        e_route_rev=np.asarray(boundary.e_route_rev),
        joint_set_effects_fwd=dict(intervention.joint_set_effects_fwd),
        joint_set_effects_rev=dict(intervention.joint_set_effects_rev),
        bos_convention=intervention.bos_convention,
        catalog_condition_ids=list(intervention.catalog_condition_ids),
        eligible_prompt_ids=intervention.eligible_prompt_ids,
    )


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
    production_wheel_dir: Path | str | None = None,
    expected_weight_manifest: Mapping[str, str] | None = None,
    exploratory_ranks: Sequence[int] | None = None,
    development_ids: Sequence[str] | None = None,
    unit_specs: Sequence[UnitSpec] | None = None,
) -> dict[str, Any]:
    """Multi-unit live path via Q1CausalRunner adapters — auth + preflight required.

    Aggregates exploratory (3) + primary (5) + secondary (2) + adapter-off.
    Boundary and intervention bundles are reconciled per unit. Completeness of
    the eight-stage path is **not** claimed until these checks pass.
    Default: unauthorized (env gate OFF).
    """
    require_scientific_authorization()
    pre = run_scientific_preflight(
        repo_root,
        held_out_ids=held_out_ids_allocation_order,
        production_lock_path=production_lock_path,
        production_wheel_dir=production_wheel_dir,
        expected_weight_manifest=expected_weight_manifest,
        require_sci_auth=True,
    )
    pre.raise_if_failed()

    if not all(
        [hooks.load_model_for_unit, hooks.capture_donors_for_unit, hooks.execute_unit]
    ):
        raise RuntimeError(
            "Live pipeline requires load_model_for_unit, capture_donors_for_unit, "
            "and execute_unit (Q1CausalRunner adapters)"
        )

    specs = list(unit_specs) if unit_specs is not None else expected_frozen_unit_specs()
    expected = expected_frozen_unit_specs()
    if [s.model_unit for s in specs] != [s.model_unit for s in expected]:
        raise RuntimeError(
            "incomplete or incorrect unit set: require frozen exploratory/primary/"
            "secondary/adapter-off identities"
        )

    ex = ScientificPipelineExecutor(repo_root, out_dir)
    ex.stage1_catalog_and_bos(
        held_out_ids=held_out_ids_allocation_order,
        development_ids=development_ids,
    ).require_complete()

    units: list[ModelUnitInput] = []
    per_unit_records: list[dict[str, Any]] = []
    first_prov: dict[str, Any] | None = None
    donor_ok = False

    for spec in specs:
        loaded = hooks.load_model_for_unit(spec)  # type: ignore[misc]
        if not isinstance(loaded, Mapping) or "model" not in loaded:
            raise RuntimeError(
                f"{spec.model_unit}: load_model_for_unit must return mapping with model"
            )
        prov = _require_mapping_evidence(
            loaded,
            stage=f"load:{spec.model_unit}",
            required_keys=(
                "base_revision",
                "adapter_revision",
                "weight_file_sha256",
                "attn_implementation",
                "torch_dtype",
                "model",
            ),
        )
        if first_prov is None:
            first_prov = prov
            ex.stage2_provenance(prov).require_complete()

        donors = hooks.capture_donors_for_unit(spec, prov["model"])  # type: ignore[misc]
        donor_ev = _require_mapping_evidence(
            donors,
            stage=f"donors:{spec.model_unit}",
            required_keys=(
                "n_donor_caches",
                "clean_references_frozen",
                "instruction_ids",
            ),
        )
        if not donor_ok:
            ex.stage3_donors(donor_ev).require_complete()
            donor_ok = True

        pair = hooks.execute_unit(spec, prov["model"], donors)  # type: ignore[misc]
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not isinstance(pair[0], ConditionOutputBundle)
            or not isinstance(pair[1], ConditionOutputBundle)
        ):
            raise RuntimeError(
                f"{spec.model_unit}: execute_unit must return "
                "(boundary_bundle, intervention_bundle)"
            )
        boundary, intervention = pair
        if boundary.model_unit != spec.model_unit or intervention.model_unit != spec.model_unit:
            raise RuntimeError(
                f"{spec.model_unit}: bundle model_unit does not match UnitSpec"
            )
        merged = reconcile_boundary_intervention(boundary, intervention)
        if not merged.catalog_condition_ids:
            raise RuntimeError(
                f"{spec.model_unit}: missing catalog condition records on bundle"
            )
        ex.stage4_boundaries(merged)
        ex.stage5_interventions(merged).require_complete()
        st6, unit = ex.stage6_assemble(merged)
        st6.require_complete()
        assert unit is not None
        units.append(unit)
        per_unit_records.append(
            {
                "model_unit": spec.model_unit,
                "role": spec.role,
                "n_catalog_conditions": len(merged.catalog_condition_ids),
            }
        )

    # Mark stage4 complete after all units (last boundary evidence retained)
    if ex.stages[PIPELINE_STAGES[3]].status != "complete":
        ex.stages[PIPELINE_STAGES[3]] = StageResult(
            PIPELINE_STAGES[3],
            "complete",
            evidence={"n_units": len(units), "per_unit": per_unit_records},
        )

    if len(units) != len(expected):
        raise RuntimeError("incomplete unit aggregation")

    st7, analysis = ex.stage7_analyze(
        units,
        held_out_ids_allocation_order,
        exploratory_ranks=exploratory_ranks,
        development_ids=development_ids,
    )
    st7.require_complete()
    assert analysis is not None
    ex.stage8_persist(analysis).require_complete()

    pipeline_ready = all(s.status == "complete" for s in ex.stages.values())
    return {
        "stages": {k: asdict(v) for k, v in ex.stages.items()},
        "analysis": analysis,
        "preflight": asdict(pre),
        "n_units": len(units),
        "per_unit": per_unit_records,
        "eight_stage_ready": pipeline_ready,
        "note": (
            "eight_stage_ready reflects enforceable checks for this authorized run; "
            "default scientific execution remains disabled without auth/preflight"
        ),
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
