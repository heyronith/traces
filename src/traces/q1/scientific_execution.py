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
from traces.q1.analysis import exploratory_stop_triggered, rank_pairs_signed
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS, validate_pair_effect_map
from traces.q1.frozen_artifacts import (
    REQUIRED_EXPLORATORY_SEEDS,
    coerce_frozen_dolly_id,
    ordered_id_sequence_sha256,
    verify_frozen_artifacts,
    verify_runtime_allocation_ids,
)
from traces.q1.patching import NOMINATED_PAIR
from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
from traces.q1.runtime_lock import claim_production_lock_exists, load_lock_document
from traces.q1.runtime_spec import (
    DEFAULT_BATCHING,
    DEFAULT_DETERMINISM,
    DIRECT_PINS_SHA256,
    REQUIRED_ATTN_IMPLEMENTATION,
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


def require_tiny_gemma_authorization() -> None:
    """Authorize randomly-initialized tiny-Gemma engineering forwards only.

    Does **not** authorize scientific execution. The tiny env flag alone never
    enables ``TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION``.
    """
    if os.environ.get(TINY_AUTH_ENV) != "1":
        raise PermissionError(
            f"Tiny-Gemma forward refused: set {TINY_AUTH_ENV}=1 for engineering tests. "
            f"This does not authorize scientific execution ({SCI_AUTH_ENV})."
        )


def require_causal_runner_authorization(
    *,
    engineering_tiny: bool = False,
    model: Any = None,
) -> None:
    """Fail-closed auth for Q1CausalRunner forwards.

    Scientific production path requires ``TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION=1``.
    Tiny path requires the separate tiny env flag **and** a verified engineering
    fixture (caller boolean alone is insufficient).
    """
    if engineering_tiny:
        require_tiny_gemma_authorization()
        if model is None:
            raise PermissionError(
                "engineering_tiny path requires the model object for fixture verification"
            )
        from traces.q1.engineering_fixtures import require_verified_engineering_tiny

        require_verified_engineering_tiny(model)
        # Tiny flag must never satisfy scientific auth
        if scientific_auth_status().authorized and os.environ.get(SCI_AUTH_ENV) != "1":
            raise PermissionError("internal auth inconsistency")
        return
    require_scientific_authorization()


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
    # Descriptive downstream secondary head effects (D1) — never mixed into
    # head_effects_fwd/rev used for H2/H3/crossfit (upstream 152 only).
    downstream_head_effects_fwd: dict[tuple[int, int], np.ndarray] = field(
        default_factory=dict
    )
    downstream_head_effects_rev: dict[tuple[int, int], np.ndarray] = field(
        default_factory=dict
    )

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

        adapter_off = (
            self.family == "base" or self.role == "adapter_off_gates_descriptives"
        )
        for label, fx in (
            ("pair_fwd", self.pair_effects_fwd),
            ("pair_rev", self.pair_effects_rev),
            ("head_fwd", self.head_effects_fwd),
            ("head_rev", self.head_effects_rev),
        ):
            if not isinstance(fx, Mapping):
                failures.append(f"{label}: not a mapping")
                continue
            # Adapter-off baseline intentionally has empty intervention maps (B1-C5)
            if not fx:
                if not adapter_off:
                    failures.append(f"{label}: empty or not a mapping")
                continue
            for k, v in fx.items():
                _check_vec(f"{label}{k}", v)

        for label, fx in (
            ("joint_fwd", self.joint_set_effects_fwd),
            ("joint_rev", self.joint_set_effects_rev),
        ):
            for k, v in fx.items():
                a = np.asarray(v, dtype=np.float64)
                if a.ndim != 1 or len(a) != expected_len:
                    failures.append(
                        f"{label}{k}: length/shape mismatch vs n_eligible {expected_len}"
                    )
                    continue
                # Cross-fit fold-eval arrays may be partially NaN (unmeasured fold)
                if not np.any(np.isfinite(a)):
                    failures.append(f"{label}{k}: no finite measured effects")

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
    held_out_ids: Sequence[Any] | None = None,
    development_ids: Sequence[Any] | None = None,
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

    if held_out_ids is not None or development_ids is not None:
        alloc = verify_runtime_allocation_ids(
            root, held_out_ids=held_out_ids, development_ids=development_ids
        )
        checks["allocation_binding"] = alloc
        if not alloc["ok"]:
            failures.extend(alloc["failures"])

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
    held_out_ids: Sequence[Any],
    *,
    expected: int = N_HO,
    repo_root: Path | str | None = None,
    development_ids: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Size + fold structure; when ``repo_root`` given, bind to frozen dolly_ids."""
    enforce_production_held_out_size(len(held_out_ids), expected=expected)
    folds = assign_f2_folds(list(held_out_ids))
    out: dict[str, Any] = {
        "n": len(held_out_ids),
        "fold_a_n": len(folds.fold_a_ids),
        "fold_b_n": len(folds.fold_b_ids),
        "allocation_order_preserved": True,
    }
    if repo_root is not None:
        bound = verify_runtime_allocation_ids(
            repo_root,
            held_out_ids=held_out_ids,
            development_ids=development_ids,
        )
        out["frozen_binding"] = bound
        if not bound["ok"]:
            raise RuntimeError(
                "held-out/development IDs do not match frozen allocation: "
                + "; ".join(bound["failures"])
            )
    return out


# Frozen nominated pair as (L15 head index, L16 head index) among 64 comparators
NOMINATED_PAIR_KEY: tuple[int, int] = (
    NOMINATED_PAIR[0].head_index,
    NOMINATED_PAIR[1].head_index,
)
assert NOMINATED_PAIR_KEY == (6, 5)
assert NOMINATED_PAIR_KEY in EXPECTED_L15_L16_PAIRS


def compute_exploratory_rank_from_unit(
    unit: ModelUnitInput,
    *,
    expected_seed: int,
    development_ids: Sequence[Any],
) -> dict[str, Any]:
    """Compute nominated-pair rank from completed exploratory intervention maps.

    Requires: l1523 exploratory identity, single-BOS, development-dataset ID
    binding, and the complete 64-pair comparator population.
    """
    failures: list[str] = []
    if unit.family != "l1523":
        failures.append(f"family {unit.family!r} != l1523")
    if unit.role != "exploratory_dev":
        failures.append(f"role {unit.role!r} != exploratory_dev")
    # Seed from model_unit name …_seedN
    seed_from_name: int | None = None
    if "_seed" in unit.model_unit:
        try:
            seed_from_name = int(unit.model_unit.rsplit("_seed", 1)[-1])
        except ValueError:
            seed_from_name = None
    if seed_from_name != expected_seed:
        failures.append(
            f"seed identity {seed_from_name!r} != expected {expected_seed}"
        )
    if getattr(unit, "bos_convention", "single") not in (None, "single"):
        # ModelUnitInput may not carry bos; ConditionOutputBundle enforces single
        failures.append("exploratory stop ranks require single-BOS convention")

    # Dataset identity: prompt_ids must match frozen development dolly_ids
    try:
        got_ids = [coerce_frozen_dolly_id(x) for x in unit.prompt_ids]
    except ValueError as e:
        failures.append(f"prompt_ids not frozen dolly_ids: {e}")
        got_ids = []
    try:
        expected_dev = [coerce_frozen_dolly_id(x) for x in development_ids]
    except ValueError as e:
        failures.append(f"development_ids invalid: {e}")
        expected_dev = []
    if got_ids != expected_dev:
        failures.append(
            "exploratory dataset identity mismatch vs frozen development allocation"
        )
    if len(expected_dev) != N_DEV:
        failures.append(f"development allocation n={len(expected_dev)} != {N_DEV}")

    n_elig = int(np.asarray(unit.eligible_mask, dtype=bool).sum())
    pair_chk = validate_pair_effect_map(
        unit.pair_effects_fwd, n_eligible=n_elig, require_complete_64=True
    )
    if not pair_chk["ok"]:
        failures.extend(pair_chk["failures"])

    rank: int | None = None
    if not failures and NOMINATED_PAIR_KEY in unit.pair_effects_fwd:
        means = {
            p: float(np.mean(np.asarray(v, dtype=np.float64)))
            for p, v in unit.pair_effects_fwd.items()
        }
        ranking = rank_pairs_signed(means)
        rank_map = {p: r for p, _, r in ranking}
        rank = int(rank_map[NOMINATED_PAIR_KEY])
    elif NOMINATED_PAIR_KEY not in unit.pair_effects_fwd:
        failures.append(f"nominated pair {NOMINATED_PAIR_KEY} missing from pair map")

    return {
        "ok": len(failures) == 0 and rank is not None,
        "failures": failures,
        "seed": expected_seed,
        "model_unit": unit.model_unit,
        "rank": rank,
        "nominated_pair": NOMINATED_PAIR_KEY,
        "n_comparator_pairs": len(unit.pair_effects_fwd),
        "bos_convention": "single",
        "development_ids_sha256": ordered_id_sequence_sha256(expected_dev)
        if expected_dev
        else None,
        "source": "computed_from_exploratory_intervention_records",
    }


def bind_exploratory_ranks_from_results(
    exploratory_units: Sequence[ModelUnitInput],
    *,
    exploratory_specs: Sequence[UnitSpec],
    development_ids: Sequence[Any] | None,
    caller_ranks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Bind stop ranks to completed exploratory results; reject unbound callers.

    Caller-supplied ranks are never used as the source of truth. If provided,
    they must exactly match the independently computed ranks or the gate blocks.
    """
    seeds = [s.seed for s in exploratory_specs]
    seeds_ok = (
        list(seeds) == list(REQUIRED_EXPLORATORY_SEEDS)
        and all(s.family == "l1523" for s in exploratory_specs)
        and all(s.role == "exploratory_dev" for s in exploratory_specs)
        and len(exploratory_units) == 3
    )
    if development_ids is None:
        return {
            "ok": False,
            "confirmatory_blocked": True,
            "triggered": False,
            "ranks": None,
            "seeds": seeds,
            "fabricated_ranks": False,
            "caller_ranks_rejected": caller_ranks is not None,
            "records": [],
            "reason": "development_ids required to bind exploratory rank dataset identity",
        }
    if not seeds_ok:
        return {
            "ok": False,
            "confirmatory_blocked": True,
            "triggered": False,
            "ranks": None,
            "seeds": seeds,
            "fabricated_ranks": False,
            "caller_ranks_rejected": False,
            "records": [],
            "reason": (
                "exploratory gate incomplete: require l1523 seeds "
                f"{list(REQUIRED_EXPLORATORY_SEEDS)} with completed units"
            ),
        }

    records: list[dict[str, Any]] = []
    failures: list[str] = []
    for spec, unit in zip(exploratory_specs, exploratory_units):
        assert spec.seed is not None
        rec = compute_exploratory_rank_from_unit(
            unit, expected_seed=int(spec.seed), development_ids=development_ids
        )
        records.append(rec)
        if not rec["ok"]:
            failures.extend(
                [f"seed{spec.seed}: {f}" for f in rec["failures"]]
                or [f"seed{spec.seed}: rank computation failed"]
            )

    ranks = [int(r["rank"]) for r in records if r.get("rank") is not None]
    ranks_ok = len(ranks) == 3 and len(failures) == 0

    caller_rejected = False
    caller_mismatch = False
    if caller_ranks is not None:
        # Caller ranks lack independent binding unless they match computed records
        if not ranks_ok:
            caller_rejected = True
            failures.append(
                "caller-supplied ranks rejected: exploratory result ranks incomplete"
            )
        else:
            got_caller = [int(x) for x in caller_ranks]
            if got_caller != ranks:
                caller_rejected = True
                caller_mismatch = True
                failures.append(
                    f"caller-supplied ranks {got_caller} disagree with "
                    f"bound exploratory result ranks {ranks}"
                )

    if not ranks_ok or caller_mismatch:
        return {
            "ok": False,
            "confirmatory_blocked": True,
            "triggered": False,
            "ranks": ranks if len(ranks) == 3 else None,
            "seeds": seeds,
            "fabricated_ranks": False,
            "caller_ranks_rejected": caller_rejected,
            "caller_ranks": list(caller_ranks) if caller_ranks is not None else None,
            "records": records,
            "reason": "; ".join(failures) if failures else "bound ranks incomplete",
        }

    triggered = exploratory_stop_triggered(ranks)
    return {
        "ok": True,
        "confirmatory_blocked": triggered,
        "triggered": triggered,
        "ranks": ranks,
        "seeds": seeds,
        "fabricated_ranks": False,
        "caller_ranks_rejected": False,
        "caller_ranks": list(caller_ranks) if caller_ranks is not None else None,
        "records": records,
        "criterion": ">=2 of 3 ranks worse than 10",
        "reason": "stop triggered" if triggered else "exploratory clear",
        "nominated_pair": NOMINATED_PAIR_KEY,
        "comparator_population": 64,
        "bos_convention": "single",
    }


def evaluate_live_exploratory_gate(
    *,
    exploratory_specs: Sequence[UnitSpec],
    exploratory_units: Sequence[ModelUnitInput] | None = None,
    development_ids: Sequence[Any] | None = None,
    exploratory_ranks: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate frozen exploratory stop **before** confirmatory model callbacks.

    Ranks must be computed from completed exploratory intervention records for
    seeds 42–44. Unbound caller-supplied ranks are rejected.
    """
    if exploratory_units is None:
        # No result records → cannot bind ranks; never fabricate
        return {
            "triggered": False,
            "confirmatory_blocked": True,
            "ranks": None,
            "seeds": [s.seed for s in exploratory_specs],
            "fabricated_ranks": False,
            "caller_ranks_rejected": exploratory_ranks is not None,
            "reason": (
                "exploratory result records required to bind stop ranks; "
                "caller-supplied ranks alone are insufficient"
            ),
        }
    return bind_exploratory_ranks_from_results(
        exploratory_units,
        exploratory_specs=exploratory_specs,
        development_ids=development_ids,
        caller_ranks=exploratory_ranks,
    )


def evaluate_live_integrity_battery(
    *,
    exploratory_units: Sequence[ModelUnitInput],
    development_ids: Sequence[Any] | None = None,
    integrity_evidence: Mapping[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Full exploratory integrity gate battery (authenticated empirical artifacts).

    Developmental eligibility / first-token gates are **derived from exploratory
    result records** — external summaries cannot override them. Empirical
    artifacts must pass provenance + frozen bindings; synthetic fixtures never
    clear production progression. Manifest authentication uses the fixed repo
    path under ``repo_root`` (never an artifact-controlled path).
    """
    from traces.q1.frozen_artifacts import (
        FROZEN_DEV_ID_SEQUENCE_SHA256,
        ordered_id_sequence_sha256,
    )
    from traces.q1.integrity_gates import evaluate_preconfirmation_gates

    ev = dict(integrity_evidence or {})
    extra_failures: list[str] = []

    # Frozen development denominator from allocation IDs when provided
    if development_ids is not None:
        n_frozen = len(list(development_ids))
        try:
            seq_sha = ordered_id_sequence_sha256([int(x) for x in development_ids])
        except (TypeError, ValueError):
            seq_sha = None
        if seq_sha is not None and seq_sha != FROZEN_DEV_ID_SEQUENCE_SHA256:
            extra_failures.append(
                "development_ids sequence SHA != FROZEN_DEV_ID_SEQUENCE_SHA256"
            )
    else:
        n_frozen = int(exploratory_units[0].n_frozen) if exploratory_units else N_DEV

    # Derive developmental gates strictly from exploratory records
    if not exploratory_units:
        development_n_eligible = None
        first_token = None
        extra_failures.append(
            "exploratory result records required to derive eligibility/first-token"
        )
    else:
        development_n_eligible = min(
            int(u.eligible_mask.sum()) for u in exploratory_units
        )
        first_token = min(float(u.first_token_validity) for u in exploratory_units)
        # External summaries may be checked but never override derived values
        _ = ev.get("development_n_eligible")
        _ = ev.get("exploratory_first_token_validity")

    identity_ref = ev.get("identity_artifact") or ev.get("identity_suite")
    f4_ref = ev.get("f4_artifact") or ev.get("f4_table")
    double_ref = ev.get("double_bos_artifact") or ev.get("double_bos_evidence")

    # Manifest path derives exclusively from the trusted executor repo_root.
    # Evidence-supplied repo_root / test-trust overrides are ignored.
    _ = ev.get("repo_root")
    _ = ev.get("_test_pinned_digest")
    _ = ev.get("_allow_synthetic_for_test_battery")
    root = repo_root
    if root is None:
        extra_failures.append(
            "trusted executor repo_root required "
            "(integrity_evidence repo_root is ignored)"
        )
    frozen_prompts = (
        [str(x) for x in development_ids] if development_ids is not None else None
    )

    # Strip test-trust keys from artifact refs before the production gate
    def _scrub(ref: Any) -> Any:
        if isinstance(ref, Mapping):
            return {
                k: v
                for k, v in ref.items()
                if k
                not in {
                    "_test_pinned_digest",
                    "_allow_synthetic_for_test_battery",
                    "repo_root",
                }
            }
        return ref

    identity_ref = _scrub(identity_ref)
    f4_ref = _scrub(f4_ref)
    double_ref = _scrub(double_ref)

    report = evaluate_preconfirmation_gates(
        development_n_eligible=(
            int(development_n_eligible) if development_n_eligible is not None else None
        ),
        development_n_frozen=int(n_frozen),
        exploratory_first_token_validity=(
            float(first_token) if first_token is not None else None
        ),
        identity_suite=identity_ref if isinstance(identity_ref, Mapping) else None,
        f4_table=f4_ref if isinstance(f4_ref, Mapping) else None,
        double_bos_scan_completed=bool(ev.get("double_bos_scan_completed", False)),
        double_bos_n_pairs=ev.get("double_bos_n_pairs"),
        double_bos_evidence=double_ref if isinstance(double_ref, Mapping) else None,
        require_confirmatory=False,
        require_empirical_scientific=True,
        repo_root=str(root) if root else None,
        frozen_prompt_ids=frozen_prompts,
    )
    blocked_reasons = list(report.blocked_reasons) + extra_failures
    progression_blocked = bool(report.progression_blocked) or bool(extra_failures)
    return {
        "progression_blocked": progression_blocked,
        "blocked_reasons": blocked_reasons,
        "gates": report.to_dict()["gates"],
        "confirmatory_blocked": progression_blocked,
        "developmental_gates_source": "exploratory_result_records",
        "reason": (
            "integrity battery blocked: " + "; ".join(blocked_reasons)
            if progression_blocked
            else "integrity battery clear"
        ),
    }


def evaluate_per_model_confirmatory_validity(
    units: Sequence[ModelUnitInput],
) -> dict[str, Any]:
    """Per-model §6 validity before confirmatory analysis."""
    from traces.q1.integrity_gates import gate_confirmatory_model_validity

    per_unit: list[dict[str, Any]] = []
    blocked = False
    reasons: list[str] = []
    for u in units:
        if u.role not in CONFIRMATORY_ROLES:
            continue
        g = gate_confirmatory_model_validity(
            trigger_asr=float(u.trigger_asr),
            clean_false_fire=float(u.clean_false_fire),
            first_token_validity=float(u.first_token_validity),
            n_eligible=int(u.eligible_mask.sum()),
            n_frozen=int(u.n_frozen),
        )
        entry = {
            "model_unit": u.model_unit,
            "role": u.role,
            "status": g.status,
            "failures": list(g.failures),
            "evidence": dict(g.evidence),
        }
        per_unit.append(entry)
        if not g.ok:
            blocked = True
            reasons.append(f"{u.model_unit}:{g.status}")
            reasons.extend(g.failures)
    return {
        "analysis_blocked": blocked,
        "blocked_reasons": reasons,
        "per_unit": per_unit,
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


CONFIRMATORY_ROLES = frozenset({"primary_confirmatory", "secondary_replication"})


def verify_unit_load_provenance(
    spec: UnitSpec,
    prov: Mapping[str, Any],
) -> dict[str, Any]:
    """Per-unit provenance — never reuse another unit's validation result."""
    failures: list[str] = []
    for field in ("model_unit", "family", "role", "seed"):
        if field not in prov:
            failures.append(f"missing provenance field {field}")
        elif prov[field] != getattr(spec, field):
            failures.append(
                f"{field} {prov.get(field)!r} != spec {getattr(spec, field)!r}"
            )
    if prov.get("base_revision") != PINNED_BASE_REVISION:
        failures.append(f"base_revision {prov.get('base_revision')!r} != pin")
    if spec.family != "base":
        if prov.get("adapter_revision") != PINNED_ADAPTER_REVISION:
            failures.append(
                f"adapter_revision {prov.get('adapter_revision')!r} != pin"
            )
    if prov.get("attn_implementation") != REQUIRED_ATTN_IMPLEMENTATION:
        failures.append("attn_implementation != eager")
    dtype = str(prov.get("torch_dtype", "")).lower()
    if "bfloat16" not in dtype and "bf16" not in dtype:
        failures.append("torch_dtype not bfloat16")
    weights = prov.get("weight_file_sha256")
    if not isinstance(weights, Mapping) or not weights:
        failures.append("weight_file_sha256 missing for unit")
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "model_unit": spec.model_unit,
        "family": spec.family,
        "role": spec.role,
        "seed": spec.seed,
    }


def verify_unit_donor_evidence(
    spec: UnitSpec, donors: Mapping[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    for key in ("n_donor_caches", "clean_references_frozen", "instruction_ids"):
        if key not in donors:
            failures.append(f"donor evidence missing {key}")
    if donors.get("clean_references_frozen") is not True:
        failures.append("clean_references_frozen must be True")
    n = donors.get("n_donor_caches")
    if not isinstance(n, (int, np.integer)) or int(n) < 1:
        failures.append("n_donor_caches must be positive int")
    ids = donors.get("instruction_ids")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or len(ids) < 1:
        failures.append("instruction_ids must be non-empty sequence")
    return {
        "ok": len(failures) == 0,
        "failures": failures,
        "model_unit": spec.model_unit,
    }


# Required keys for the live load → donors → execute handoff (B1-C9).
LOAD_CONTEXT_REQUIRED_KEYS: tuple[str, ...] = (
    "model",
    "tokenizer",
    "model_unit",
    "family",
    "role",
    "seed",
    "base_revision",
    "adapter_revision",
    "weight_file_sha256",
    "attn_implementation",
    "torch_dtype",
)


@dataclass(frozen=True)
class LoadContext:
    """Typed load context passed to donor capture and unit execution (B1-C9).

    Never pass a bare model object: production adapters require tokenizer and
    unit provenance fields from the same verified load.
    """

    model: Any
    tokenizer: Any
    model_unit: str
    family: str
    role: str
    seed: int | None
    base_revision: Any
    adapter_revision: Any
    weight_file_sha256: Mapping[str, str]
    attn_implementation: str
    torch_dtype: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, Any]:
        out = {
            "model": self.model,
            "tokenizer": self.tokenizer,
            "model_unit": self.model_unit,
            "family": self.family,
            "role": self.role,
            "seed": self.seed,
            "base_revision": self.base_revision,
            "adapter_revision": self.adapter_revision,
            "weight_file_sha256": dict(self.weight_file_sha256),
            "attn_implementation": self.attn_implementation,
            "torch_dtype": self.torch_dtype,
        }
        out.update(dict(self.extra))
        return out


def require_load_context(
    loaded: Any, *, stage: str, spec: UnitSpec | None = None
) -> dict[str, Any]:
    """Fail closed unless ``loaded`` is a full load-context mapping (B1-C9).

    Structural keys + non-None model/tokenizer are required here. Unit identity
    matching against ``spec`` is enforced by ``verify_unit_load_provenance``.
    """
    _ = spec  # reserved for stage naming / future strict binding
    if not isinstance(loaded, Mapping):
        raise RuntimeError(
            f"{stage}: load context must be a mapping with model+tokenizer; "
            f"got {type(loaded).__name__} (bare model objects are refused)"
        )
    missing = [k for k in LOAD_CONTEXT_REQUIRED_KEYS if k not in loaded]
    if missing:
        raise RuntimeError(f"{stage}: load context missing keys {missing}")
    if loaded.get("tokenizer") is None:
        raise RuntimeError(f"{stage}: load context tokenizer is required")
    if loaded.get("model") is None:
        raise RuntimeError(f"{stage}: load context model is required")
    return dict(loaded)


@dataclass
class LiveForwardHooks:
    """Typed Q1CausalRunner adapters for authorized live execution.

    ``load_model_for_unit`` returns a load-context mapping (model + tokenizer +
    provenance). ``capture_donors_for_unit`` and ``execute_unit`` receive that
    same load context — never a bare model object (B1-C9).
    """

    load_model_for_unit: Callable[[UnitSpec], Mapping[str, Any]] | None = None
    capture_donors_for_unit: Callable[[UnitSpec, Mapping[str, Any]], Mapping[str, Any]] | None = None
    execute_unit: Callable[
        [UnitSpec, Mapping[str, Any], Any],
        tuple[ConditionOutputBundle, ConditionOutputBundle],
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
        downstream_head_effects_fwd=dict(intervention.downstream_head_effects_fwd),
        downstream_head_effects_rev=dict(intervention.downstream_head_effects_rev),
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
        self,
        *,
        held_out_ids: Sequence[Any],
        development_ids: Sequence[Any] | None = None,
        bind_frozen_allocation: bool = False,
    ) -> StageResult:
        catalog = enumerate_primary_l19_conditions(include_downstream56=True)
        if not catalog:
            st = StageResult(
                PIPELINE_STAGES[0], "failed", error="empty primary catalog"
            )
        else:
            alloc: dict[str, Any] = {
                "held_out_n": len(held_out_ids),
                "held_out_expected": N_HO,
                "dev_n": len(development_ids or []),
                "dev_expected": N_DEV,
                "folds": asdict(assign_f2_folds(list(held_out_ids)))
                if len(held_out_ids) == N_HO
                else {"note": "non-production size (synthetic)"},
            }
            if bind_frozen_allocation:
                bound = verify_runtime_allocation_ids(
                    self.repo_root,
                    held_out_ids=held_out_ids,
                    development_ids=development_ids,
                )
                alloc["frozen_binding"] = bound
                if not bound["ok"]:
                    st = StageResult(
                        PIPELINE_STAGES[0],
                        "failed",
                        error="; ".join(bound["failures"]),
                        evidence={"allocation": alloc},
                    )
                    self.stages[PIPELINE_STAGES[0]] = st
                    return st
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
        adapter_off = (
            bundle.family == "base" or bundle.role == "adapter_off_gates_descriptives"
        )
        if adapter_off:
            # Descriptive adapter-off baseline: intact margins + ASR only (B1-C5)
            ok = (
                len(bundle.prompt_ids) > 0
                and len(np.asarray(bundle.m_TT)) > 0
                and len(bundle.catalog_condition_ids) > 0
            )
            err = None if ok else "adapter-off baseline margins incomplete"
        else:
            ok = (
                len(bundle.head_effects_fwd) > 0
                and len(bundle.pair_effects_fwd) > 0
                and len(bundle.joint_set_effects_fwd) > 0
            )
            err = None if ok else "intervention effect maps incomplete"
        st = StageResult(
            PIPELINE_STAGES[4],
            "complete" if ok else "failed",
            evidence={
                "model_unit": bundle.model_unit,
                "n_heads_fwd": len(bundle.head_effects_fwd),
                "n_pairs_fwd": len(bundle.pair_effects_fwd),
                "n_joint_fwd": len(bundle.joint_set_effects_fwd),
                "catalog_ids_recorded": len(bundle.catalog_condition_ids),
                "adapter_off_baseline": adapter_off,
            },
            error=err,
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
    exploratory_ranks: Sequence[int] | None = None,
    development_ids: Sequence[str] | None = None,
    skip_live_preflight: bool = True,
) -> dict[str, Any]:
    """Run stages 1–8 using typed bundles (no model forwards).

    Exploratory ranks default to ``None`` (M2): never fabricate clear ranks.
    Callers must supply evidence-bound ranks or accept confirmatory blocking.

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


def _execute_live_unit(
    *,
    ex: ScientificPipelineExecutor,
    hooks: LiveForwardHooks,
    spec: UnitSpec,
    per_unit_audits: list[dict[str, Any]],
) -> ModelUnitInput:
    """Load → per-unit provenance → donors → conditions for one unit.

    Never reuses another unit's provenance validation result.
    """
    loaded = hooks.load_model_for_unit(spec)  # type: ignore[misc]
    # B1-C9: full load context (model + tokenizer + provenance) — not bare model
    load_ctx = require_load_context(
        loaded, stage=f"load:{spec.model_unit}", spec=spec
    )
    unit_prov = verify_unit_load_provenance(spec, load_ctx)
    if not unit_prov["ok"]:
        raise RuntimeError(
            f"{spec.model_unit}: provenance failed: "
            + "; ".join(unit_prov["failures"])
        )

    donors = hooks.capture_donors_for_unit(spec, load_ctx)  # type: ignore[misc]
    donor_ev = _require_mapping_evidence(
        donors,
        stage=f"donors:{spec.model_unit}",
        required_keys=(
            "n_donor_caches",
            "clean_references_frozen",
            "instruction_ids",
        ),
    )
    donor_check = verify_unit_donor_evidence(spec, donor_ev)
    if not donor_check["ok"]:
        raise RuntimeError(
            f"{spec.model_unit}: donor validation failed: "
            + "; ".join(donor_check["failures"])
        )

    pair = hooks.execute_unit(spec, load_ctx, donors)  # type: ignore[misc]
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
    if merged.family != spec.family or merged.role != spec.role:
        raise RuntimeError(
            f"{spec.model_unit}: condition bundle family/role mismatch vs UnitSpec"
        )

    ex.stage4_boundaries(merged)
    st5 = ex.stage5_interventions(merged)
    st5.require_complete()
    st6, unit = ex.stage6_assemble(merged)
    st6.require_complete()
    assert unit is not None

    # M1: never silently skip observation persistence on the production path.
    # Engineering mocks may set donors["_engineering_skip_observations"]=True
    # (production hooks never set this flag).
    obs_meta: dict[str, Any] | None = None
    raw_conditions = donors.get("_condition_records") if isinstance(donors, Mapping) else None
    raw_f3 = donors.get("_f3_outcomes") if isinstance(donors, Mapping) else None
    eng_skip = bool(
        isinstance(donors, Mapping) and donors.get("_engineering_skip_observations") is True
    )
    if eng_skip:
        if raw_conditions is not None or raw_f3 is not None:
            raise RuntimeError(
                f"{spec.model_unit}: contradictory observation flags "
                "(_engineering_skip_observations with stashed records)"
            )
        obs_meta = None
    else:
        if raw_conditions is None or raw_f3 is None:
            raise RuntimeError(
                f"{spec.model_unit}: live execution requires raw condition and F3 "
                "generation records — refuse silent skip of observation "
                "persistence (M1)"
            )
        from traces.q1.scientific_observations import persist_unit_scientific_observations

        gen_recs = list(getattr(raw_f3, "trigger_records", []) or []) + list(
            getattr(raw_f3, "clean_records", []) or []
        )
        if not list(raw_conditions):
            raise RuntimeError(
                f"{spec.model_unit}: empty condition records — refuse completion "
                "without scientific observations (M1)"
            )
        if not gen_recs:
            raise RuntimeError(
                f"{spec.model_unit}: empty F3 generation records — refuse "
                "completion without scientific observations (M1)"
            )
        obs_meta = persist_unit_scientific_observations(
            ex.out_dir,
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            condition_records=list(raw_conditions),
            generation_records=gen_recs,
            load_provenance={
                k: load_ctx[k]
                for k in (
                    "model_unit",
                    "family",
                    "role",
                    "seed",
                    "base_revision",
                    "adapter_revision",
                    "weight_file_sha256",
                    "attn_implementation",
                    "torch_dtype",
                )
                if k in load_ctx
            },
        )

    audit = {
        "model_unit": spec.model_unit,
        "family": spec.family,
        "role": spec.role,
        "seed": spec.seed,
        "provenance": unit_prov,
        "donors": donor_check,
        "n_catalog_conditions": len(merged.catalog_condition_ids),
        "condition_validation": "ok",
        "weight_file_sha256": dict(load_ctx["weight_file_sha256"]),
        "base_revision": load_ctx["base_revision"],
        "adapter_revision": load_ctx.get("adapter_revision"),
        "attn_implementation": load_ctx["attn_implementation"],
        "torch_dtype": load_ctx["torch_dtype"],
        "observations": obs_meta,
    }
    per_unit_audits.append(audit)
    return unit


def run_live_scientific_pipeline(
    *,
    repo_root: Path | str,
    held_out_ids_allocation_order: Sequence[Any],
    hooks: LiveForwardHooks,
    out_dir: Path | str,
    production_lock_path: Path | str | None = None,
    production_wheel_dir: Path | str | None = None,
    expected_weight_manifest: Mapping[str, str] | None = None,
    exploratory_ranks: Sequence[int] | None = None,
    development_ids: Sequence[Any] | None = None,
    unit_specs: Sequence[UnitSpec] | None = None,
    integrity_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Multi-unit live path via Q1CausalRunner adapters — auth + preflight required.

    Exploratory seeds 42–44 are loaded/executed first. Primary/secondary
    confirmatory model callbacks run only after the frozen stopping rule **and**
    the full empirical integrity gate battery are clear. Missing empirical
    evidence (identities, F4, double-BOS) blocks confirmatory callbacks —
    synthetic suites / ``locked=True`` stubs / asserted scan flags do not clear.
    """
    require_scientific_authorization()
    pre = run_scientific_preflight(
        repo_root,
        held_out_ids=held_out_ids_allocation_order,
        development_ids=development_ids,
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

    # Bind IDs to frozen allocation before any model callback
    verify_held_out_allocation(
        held_out_ids_allocation_order,
        repo_root=repo_root,
        development_ids=development_ids,
    )

    specs = list(unit_specs) if unit_specs is not None else expected_frozen_unit_specs()
    expected = expected_frozen_unit_specs()
    if [s.model_unit for s in specs] != [s.model_unit for s in expected]:
        raise RuntimeError(
            "incomplete or incorrect unit set: require frozen exploratory/primary/"
            "secondary/adapter-off identities"
        )

    exploratory_specs = [s for s in specs if s.role == "exploratory_dev"]
    confirmatory_specs = [s for s in specs if s.role in CONFIRMATORY_ROLES]
    other_specs = [
        s for s in specs if s.role not in CONFIRMATORY_ROLES and s.role != "exploratory_dev"
    ]

    ex = ScientificPipelineExecutor(repo_root, out_dir)
    ex.stage1_catalog_and_bos(
        held_out_ids=held_out_ids_allocation_order,
        development_ids=development_ids,
        bind_frozen_allocation=True,
    ).require_complete()

    units: list[ModelUnitInput] = []
    per_unit_audits: list[dict[str, Any]] = []

    # --- Exploratory first (before any confirmatory load) ---
    for spec in exploratory_specs:
        units.append(
            _execute_live_unit(
                ex=ex, hooks=hooks, spec=spec, per_unit_audits=per_unit_audits
            )
        )

    # Stage 2/3 evidence: complete audit across exploratory units so far
    ex.stages[PIPELINE_STAGES[1]] = StageResult(
        PIPELINE_STAGES[1],
        "complete",
        evidence={
            "n_units_audited": len(per_unit_audits),
            "per_unit_provenance": [
                {
                    "model_unit": a["model_unit"],
                    "ok": a["provenance"]["ok"],
                    "base_revision": a["base_revision"],
                    "adapter_revision": a["adapter_revision"],
                }
                for a in per_unit_audits
            ],
            "note": "each unit validated independently; first-unit result not reused",
        },
    )
    ex.stages[PIPELINE_STAGES[2]] = StageResult(
        PIPELINE_STAGES[2],
        "complete",
        evidence={
            "n_units_with_donors": len(per_unit_audits),
            "per_unit_donors": [a["donors"] for a in per_unit_audits],
        },
    )

    gate = evaluate_live_exploratory_gate(
        exploratory_specs=exploratory_specs,
        exploratory_units=units,
        development_ids=development_ids,
        exploratory_ranks=exploratory_ranks,
    )
    bound_ranks = gate.get("ranks")

    integrity = evaluate_live_integrity_battery(
        exploratory_units=units,
        development_ids=development_ids,
        integrity_evidence=integrity_evidence,
        repo_root=repo_root,
    )

    confirmatory_blocked = bool(gate.get("confirmatory_blocked")) or bool(
        integrity.get("confirmatory_blocked")
    )
    block_reason = None
    if gate.get("confirmatory_blocked"):
        block_reason = str(gate.get("reason"))
    elif integrity.get("confirmatory_blocked"):
        block_reason = str(integrity.get("reason"))

    if confirmatory_blocked:
        # Persist exploratory-only analysis path; zero confirmatory callbacks
        st7, analysis = ex.stage7_analyze(
            units,
            [str(x) for x in held_out_ids_allocation_order],
            exploratory_ranks=bound_ranks if isinstance(bound_ranks, list) else None,
            development_ids=(
                [str(x) for x in development_ids] if development_ids is not None else None
            ),
        )
        st7.require_complete()
        assert analysis is not None
        ex.stage8_persist(analysis).require_complete()
        if ex.stages[PIPELINE_STAGES[3]].status != "complete":
            ex.stages[PIPELINE_STAGES[3]] = StageResult(
                PIPELINE_STAGES[3],
                "complete",
                evidence={"n_units": len(units), "exploratory_only": True},
            )
        return {
            "stages": {k: asdict(v) for k, v in ex.stages.items()},
            "analysis": analysis,
            "preflight": asdict(pre),
            "n_units": len(units),
            "per_unit": per_unit_audits,
            "exploratory_gate": gate,
            "integrity_gates": integrity,
            "confirmatory_executed": False,
            "eight_stage_ready": False,
            "note": (
                "confirmatory model callbacks suppressed: " + str(block_reason)
            ),
        }

    # --- Confirmatory + adapter-off only after stop + integrity clear ---
    for spec in confirmatory_specs + other_specs:
        units.append(
            _execute_live_unit(
                ex=ex, hooks=hooks, spec=spec, per_unit_audits=per_unit_audits
            )
        )

    # Refresh stage2/3 with full per-unit audits
    ex.stages[PIPELINE_STAGES[1]] = StageResult(
        PIPELINE_STAGES[1],
        "complete",
        evidence={
            "n_units_audited": len(per_unit_audits),
            "per_unit_provenance": [
                {
                    "model_unit": a["model_unit"],
                    "ok": a["provenance"]["ok"],
                    "base_revision": a["base_revision"],
                    "adapter_revision": a["adapter_revision"],
                }
                for a in per_unit_audits
            ],
            "note": "each unit validated independently; first-unit result not reused",
        },
    )
    ex.stages[PIPELINE_STAGES[2]] = StageResult(
        PIPELINE_STAGES[2],
        "complete",
        evidence={
            "n_units_with_donors": len(per_unit_audits),
            "per_unit_donors": [a["donors"] for a in per_unit_audits],
        },
    )

    if ex.stages[PIPELINE_STAGES[3]].status != "complete":
        ex.stages[PIPELINE_STAGES[3]] = StageResult(
            PIPELINE_STAGES[3],
            "complete",
            evidence={"n_units": len(units), "per_unit": per_unit_audits},
        )

    if len(units) != len(expected):
        raise RuntimeError("incomplete unit aggregation")

    # Per-model validity before confirmatory analysis (after callbacks)
    per_model = evaluate_per_model_confirmatory_validity(units)
    if per_model["analysis_blocked"]:
        # Callbacks already ran; refuse confirmatory analysis conclusions
        exploratory_only = [u for u in units if u.role == "exploratory_dev"]
        st7, analysis = ex.stage7_analyze(
            exploratory_only,
            [str(x) for x in held_out_ids_allocation_order],
            exploratory_ranks=bound_ranks if isinstance(bound_ranks, list) else None,
            development_ids=(
                [str(x) for x in development_ids] if development_ids is not None else None
            ),
        )
        st7.require_complete()
        assert analysis is not None
        analysis = dict(analysis)
        analysis["per_model_confirmatory_validity"] = per_model
        analysis["confirmatory_analysis_refused"] = True
        ex.stage8_persist(analysis).require_complete()
        return {
            "stages": {k: asdict(v) for k, v in ex.stages.items()},
            "analysis": analysis,
            "preflight": asdict(pre),
            "n_units": len(units),
            "per_unit": per_unit_audits,
            "exploratory_gate": gate,
            "integrity_gates": integrity,
            "per_model_confirmatory_validity": per_model,
            "confirmatory_executed": True,
            "confirmatory_analysis_refused": True,
            "eight_stage_ready": False,
            "note": (
                "confirmatory analysis refused: "
                + "; ".join(per_model["blocked_reasons"])
            ),
        }

    st7, analysis = ex.stage7_analyze(
        units,
        [str(x) for x in held_out_ids_allocation_order],
        exploratory_ranks=bound_ranks if isinstance(bound_ranks, list) else None,
        development_ids=(
            [str(x) for x in development_ids] if development_ids is not None else None
        ),
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
        "per_unit": per_unit_audits,
        "exploratory_gate": gate,
        "integrity_gates": integrity,
        "per_model_confirmatory_validity": per_model,
        "confirmatory_executed": True,
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
