"""Pre-confirmation integrity gates (B2).

Explicit, auditable gates that must complete before corresponding scientific
transitions. Missing evidence blocks progression. Synthetic fixtures only —
does not fabricate real checkpoint validation evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from traces.q1.conditions import enumerate_double_bos_descriptive_pairs
from traces.q1.f4 import F4CalibrationTable
from traces.q1.gates import (
    GenerationGateRates,
    confirmatory_gates_pass,
    margin_eligibility_fraction,
)
from traces.q1.identities import IdentitySuiteReport, run_synthetic_identity_suite
from traces.q1.workload import N_DEV


@dataclass
class GateEvidence:
    gate_id: str
    status: str  # complete | blocked | failed
    evidence: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "complete" and not self.failures


@dataclass
class IntegrityGateReport:
    gates: dict[str, GateEvidence]
    progression_blocked: bool
    blocked_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gates": {k: asdict(v) for k, v in self.gates.items()},
            "progression_blocked": self.progression_blocked,
            "blocked_reasons": list(self.blocked_reasons),
            "all_required_complete": not self.progression_blocked,
        }


def gate_development_ineligibility(
    *,
    n_eligible: int,
    n_frozen: int,
    max_ineligible_fraction: float = 0.10,
) -> GateEvidence:
    """Development ineligibility must not exceed 10%."""
    if n_frozen <= 0:
        return GateEvidence(
            "development_ineligibility",
            "failed",
            failures=["n_frozen must be positive"],
        )
    inelig = (n_frozen - n_eligible) / n_frozen
    ok = inelig <= max_ineligible_fraction
    return GateEvidence(
        "development_ineligibility",
        "complete" if ok else "failed",
        evidence={
            "n_eligible": n_eligible,
            "n_frozen": n_frozen,
            "ineligible_fraction": inelig,
            "max_allowed": max_ineligible_fraction,
        },
        failures=[]
        if ok
        else [f"ineligible fraction {inelig:.4f} > {max_ineligible_fraction}"],
    )


def gate_exploratory_first_token_validity(
    *,
    first_token_validity: float,
    threshold: float = 0.90,
) -> GateEvidence:
    ok = first_token_validity >= threshold
    return GateEvidence(
        "exploratory_first_token_validity",
        "complete" if ok else "failed",
        evidence={"first_token_validity": first_token_validity, "threshold": threshold},
        failures=[] if ok else [f"first_token_validity {first_token_validity} < {threshold}"],
    )


def _artifact_ref(obj: Mapping[str, Any] | None) -> tuple[str | None, str | None]:
    if obj is None:
        return None, None
    path = obj.get("artifact_path") or obj.get("path")
    sha = obj.get("sha256") or obj.get("claimed_sha256") or obj.get("persisted_sha256")
    return (str(path) if path else None, str(sha) if sha else None)


def gate_identities_1_to_9(
    suite: IdentitySuiteReport | Mapping[str, Any] | None = None,
    *,
    require_suite: bool = True,
    require_empirical_scientific: bool = False,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
    repo_root: str | None = None,
    frozen_prompt_ids: Sequence[Any] | None = None,
    f4_artifact: Mapping[str, Any] | None = None,
    sibling_artifact_digests: Mapping[str, str] | None = None,
) -> GateEvidence:
    """Identities 1–9.

    Engineering tests may use the synthetic suite when
    ``require_empirical_scientific=False``. Live scientific progression must
    supply a persisted artifact whose bytes are independently hashed — caller
    ``evidence_kind`` / ``all_passed`` metadata alone is rejected.
    """
    if require_empirical_scientific:
        from traces.q1.empirical_evidence import authenticate_identity_suite_artifact

        _ = (expected_model_units, expected_dataset_sha)  # ignored — frozen constants only
        if suite is None or isinstance(suite, IdentitySuiteReport):
            return GateEvidence(
                "identities_1_to_9",
                "blocked",
                evidence=suite.to_dict() if isinstance(suite, IdentitySuiteReport) else {},
                failures=[
                    "empirical identity evidence requires persisted artifact_path "
                    "+ claimed sha256 (synthetic suite / inline metadata insufficient)"
                ],
            )
        # Reject fabricated inline metadata even with evidence_kind set
        if "artifact_path" not in suite and "path" not in suite:
            return GateEvidence(
                "identities_1_to_9",
                "blocked",
                evidence=dict(suite),
                failures=[
                    "inline identity evidence rejected; require persisted artifact "
                    "with independently checked hash"
                ],
            )
        path, sha = _artifact_ref(suite)
        f4_path, f4_sha = _artifact_ref(f4_artifact)
        auth = authenticate_identity_suite_artifact(
            path=path or "",
            claimed_sha256=sha,
            f4_artifact_path=f4_path,
            f4_claimed_sha256=f4_sha,
            frozen_prompt_ids=frozen_prompt_ids,
            repo_root=repo_root,
            sibling_artifact_digests=sibling_artifact_digests,
        )
        return GateEvidence(
            "identities_1_to_9",
            "complete" if auth.ok else ("failed" if auth.computed_sha256 else "blocked"),
            evidence=auth.to_dict(),
            failures=list(auth.failures),
        )

    if suite is None:
        if require_suite:
            suite = run_synthetic_identity_suite()
        else:
            return GateEvidence(
                "identities_1_to_9",
                "blocked",
                failures=["identity suite evidence missing"],
            )
    if isinstance(suite, IdentitySuiteReport):
        return GateEvidence(
            "identities_1_to_9",
            "complete" if suite.all_passed else "failed",
            evidence=suite.to_dict(),
            failures=[] if suite.all_passed else ["one or more identities 1–9 failed"],
        )
    ok = bool(suite.get("all_passed"))
    return GateEvidence(
        "identities_1_to_9",
        "complete" if ok else "failed",
        evidence=dict(suite),
        failures=[] if ok else ["one or more identities 1–9 failed"],
    )


def gate_frozen_f4_tolerance_table(
    table: F4CalibrationTable | Mapping[str, Any] | None,
    *,
    require_empirical_scientific: bool = False,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
    repo_root: str | None = None,
    frozen_prompt_ids: Sequence[Any] | None = None,
    sibling_artifact_digests: Mapping[str, str] | None = None,
) -> GateEvidence:
    """Frozen F4 tolerance table must be locked with authenticated artifact.

    Caller-declared ``locked=True`` / ``evidence_kind`` / arbitrary SHA strings
    are rejected for live scientific progression.
    """
    if require_empirical_scientific:
        from traces.q1.empirical_evidence import authenticate_f4_table_artifact

        _ = (expected_model_units, expected_dataset_sha)  # ignored — frozen constants only
        if table is None or isinstance(table, F4CalibrationTable):
            return GateEvidence(
                "frozen_f4_tolerance_table",
                "blocked",
                evidence={"note": "F4 empirical lock requires authorized model repeats"},
                failures=[
                    "F4 empirical evidence requires persisted artifact_path + sha256"
                ],
            )
        if "artifact_path" not in table and "path" not in table:
            return GateEvidence(
                "frozen_f4_tolerance_table",
                "blocked",
                evidence=dict(table),
                failures=[
                    "inline F4 locked/evidence_kind metadata rejected; "
                    "require persisted artifact with independently checked hash"
                ],
            )
        path, sha = _artifact_ref(table)
        auth = authenticate_f4_table_artifact(
            path=path or "",
            claimed_sha256=sha,
            frozen_prompt_ids=frozen_prompt_ids,
            repo_root=repo_root,
            sibling_artifact_digests=sibling_artifact_digests,
        )
        return GateEvidence(
            "frozen_f4_tolerance_table",
            "complete" if auth.ok else ("failed" if auth.computed_sha256 else "blocked"),
            evidence=auth.to_dict(),
            failures=list(auth.failures),
        )

    if table is None:
        return GateEvidence(
            "frozen_f4_tolerance_table",
            "blocked",
            evidence={"note": "F4 empirical lock requires authorized model repeats"},
            failures=["F4 tolerance table evidence unavailable"],
        )
    if isinstance(table, F4CalibrationTable):
        locked = bool(table.locked)
        payload = table.to_dict()
    else:
        locked = bool(table.get("locked"))
        payload = dict(table)
    if not locked:
        return GateEvidence(
            "frozen_f4_tolerance_table",
            "blocked",
            evidence=payload,
            failures=["F4 table present but not locked"],
        )
    return GateEvidence(
        "frozen_f4_tolerance_table",
        "complete",
        evidence=payload,
    )


def gate_descriptive_double_bos_64_pair_scan(
    *,
    completed: bool = False,
    n_pairs_recorded: int | None = None,
    evidence: Mapping[str, Any] | None = None,
    require_empirical_scientific: bool = False,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
    repo_root: str | None = None,
    frozen_prompt_ids: Sequence[Any] | None = None,
    sibling_artifact_digests: Mapping[str, str] | None = None,
) -> GateEvidence:
    expected = len(enumerate_double_bos_descriptive_pairs())
    ev = dict(evidence or {})
    if require_empirical_scientific:
        from traces.q1.empirical_evidence import authenticate_double_bos_scan_artifact

        _ = (expected_model_units, expected_dataset_sha)  # ignored — frozen constants only
        if "artifact_path" not in ev and "path" not in ev:
            return GateEvidence(
                "descriptive_double_bos_64_pair_scan",
                "blocked",
                evidence={"expected_pairs": expected, **ev},
                failures=[
                    "double-BOS scan requires persisted artifact_path + sha256; "
                    "asserted completion / evidence_kind / SHA strings alone are insufficient"
                ],
            )
        path, sha = _artifact_ref(ev)
        auth = authenticate_double_bos_scan_artifact(
            path=path or "",
            claimed_sha256=sha,
            frozen_prompt_ids=frozen_prompt_ids,
            repo_root=repo_root,
            sibling_artifact_digests=sibling_artifact_digests,
        )
        return GateEvidence(
            "descriptive_double_bos_64_pair_scan",
            "complete" if auth.ok else ("failed" if auth.computed_sha256 else "blocked"),
            evidence={"expected_pairs": expected, **auth.to_dict()},
            failures=list(auth.failures),
        )
    if not completed:
        return GateEvidence(
            "descriptive_double_bos_64_pair_scan",
            "blocked",
            evidence={"expected_pairs": expected, **ev},
            failures=["descriptive double-BOS 64-pair scan not completed"],
        )
    if n_pairs_recorded is not None and n_pairs_recorded != expected:
        return GateEvidence(
            "descriptive_double_bos_64_pair_scan",
            "failed",
            evidence={"expected_pairs": expected, "n_pairs_recorded": n_pairs_recorded},
            failures=[f"recorded {n_pairs_recorded} != expected {expected} pairs"],
        )
    return GateEvidence(
        "descriptive_double_bos_64_pair_scan",
        "complete",
        evidence={"expected_pairs": expected, "n_pairs_recorded": n_pairs_recorded},
    )


def gate_confirmatory_model_validity(
    *,
    trigger_asr: float,
    clean_false_fire: float,
    first_token_validity: float,
    n_eligible: int,
    n_frozen: int,
) -> GateEvidence:
    rates = GenerationGateRates(
        trigger_asr=trigger_asr,
        clean_false_fire=clean_false_fire,
        first_token_validity=first_token_validity,
        n_prompts=n_frozen,
        n_trigger_fires=0,
        n_clean_fires=0,
        n_first_token_ok=0,
    )
    ok, failures = confirmatory_gates_pass(rates)
    frac, elig_ok = margin_eligibility_fraction(n_eligible, n_frozen)
    if not elig_ok:
        failures = list(failures) + [f"eligibility fraction {frac:.4f} < 0.90"]
        ok = False
    return GateEvidence(
        "per_model_confirmatory_validity",
        "complete" if ok else "failed",
        evidence={
            "trigger_asr": trigger_asr,
            "clean_false_fire": clean_false_fire,
            "first_token_validity": first_token_validity,
            "eligibility_fraction": frac,
        },
        failures=list(failures),
    )


def evaluate_preconfirmation_gates(
    *,
    development_n_eligible: int | None = None,
    development_n_frozen: int = N_DEV,
    exploratory_first_token_validity: float | None = None,
    identity_suite: IdentitySuiteReport | Mapping[str, Any] | None = None,
    f4_table: F4CalibrationTable | Mapping[str, Any] | None = None,
    double_bos_scan_completed: bool = False,
    double_bos_n_pairs: int | None = None,
    double_bos_evidence: Mapping[str, Any] | None = None,
    confirmatory_rates: Mapping[str, float] | None = None,
    confirmatory_n_eligible: int | None = None,
    confirmatory_n_frozen: int | None = None,
    require_confirmatory: bool = False,
    require_empirical_scientific: bool = False,
    expected_model_units: Sequence[str] | None = None,
    expected_dataset_sha: str | None = None,
    repo_root: str | None = None,
    frozen_prompt_ids: Sequence[Any] | None = None,
) -> IntegrityGateReport:
    """Run the B2 gate battery. Missing evidence → blocked progression.

    Live scientific path must set ``require_empirical_scientific=True`` so
    synthetic identity suites, bare ``locked=True``, asserted scan flags, and
    unauthenticated SHA strings cannot clear the gate.
    """
    gates: dict[str, GateEvidence] = {}

    if development_n_eligible is None:
        gates["development_ineligibility"] = GateEvidence(
            "development_ineligibility",
            "blocked",
            failures=["development eligibility evidence missing"],
        )
    else:
        gates["development_ineligibility"] = gate_development_ineligibility(
            n_eligible=development_n_eligible, n_frozen=development_n_frozen
        )

    if exploratory_first_token_validity is None:
        gates["exploratory_first_token_validity"] = GateEvidence(
            "exploratory_first_token_validity",
            "blocked",
            failures=["exploratory first-token validity evidence missing"],
        )
    else:
        gates["exploratory_first_token_validity"] = gate_exploratory_first_token_validity(
            first_token_validity=exploratory_first_token_validity
        )

    f4_ref = f4_table if isinstance(f4_table, Mapping) else None
    id_ref = identity_suite if isinstance(identity_suite, Mapping) else None
    db_ref = double_bos_evidence if isinstance(double_bos_evidence, Mapping) else None

    if require_empirical_scientific:
        from traces.q1.empirical_evidence import authenticate_empirical_scientific_battery

        f4_path, f4_sha = _artifact_ref(f4_ref)
        id_path, id_sha = _artifact_ref(id_ref)
        db_path, db_sha = _artifact_ref(db_ref)
        test_pin = None
        allow_synth = False
        if isinstance(f4_ref, Mapping):
            test_pin = f4_ref.get("_test_pinned_digest") or (
                id_ref.get("_test_pinned_digest") if isinstance(id_ref, Mapping) else None
            )
            allow_synth = bool(f4_ref.get("_allow_synthetic_for_test_battery"))
        if isinstance(id_ref, Mapping) and id_ref.get("_test_pinned_digest"):
            test_pin = id_ref.get("_test_pinned_digest")
            allow_synth = allow_synth or bool(
                id_ref.get("_allow_synthetic_for_test_battery")
            )
        # Prefer explicit battery control kwargs via double_bos evidence meta
        if isinstance(db_ref, Mapping):
            if db_ref.get("_test_pinned_digest"):
                test_pin = db_ref.get("_test_pinned_digest")
            allow_synth = allow_synth or bool(
                db_ref.get("_allow_synthetic_for_test_battery")
            )

        missing_refs: list[str] = []
        if not f4_path or not f4_sha:
            missing_refs.append("F4 artifact_path+sha256")
        if not id_path or not id_sha:
            missing_refs.append("identity artifact_path+sha256")
        if not db_path or not db_sha:
            missing_refs.append("double-BOS artifact_path+sha256")
        if missing_refs or repo_root is None or frozen_prompt_ids is None:
            fail_msg = (
                ["empirical scientific battery requires all three persisted artifacts"]
                + [f"missing {m}" for m in missing_refs]
            )
            if repo_root is None:
                fail_msg.append("trusted executor repo_root required")
            if frozen_prompt_ids is None:
                fail_msg.append("frozen_prompt_ids required")
            blocked = GateEvidence(
                "empirical_scientific_battery",
                "blocked",
                failures=fail_msg,
            )
            gates["frozen_f4_tolerance_table"] = GateEvidence(
                "frozen_f4_tolerance_table", "blocked", failures=list(fail_msg)
            )
            gates["identities_1_to_9"] = GateEvidence(
                "identities_1_to_9", "blocked", failures=list(fail_msg)
            )
            gates["descriptive_double_bos_64_pair_scan"] = GateEvidence(
                "descriptive_double_bos_64_pair_scan",
                "blocked",
                failures=list(fail_msg),
            )
            gates["empirical_scientific_battery"] = blocked
        else:
            battery = authenticate_empirical_scientific_battery(
                f4_path=f4_path,
                f4_sha256=f4_sha,
                identity_path=id_path,
                identity_sha256=id_sha,
                double_bos_path=db_path,
                double_bos_sha256=db_sha,
                repo_root=repo_root,
                frozen_prompt_ids=frozen_prompt_ids,
                test_pinned_digest=str(test_pin) if test_pin else None,
                allow_synthetic_for_test_battery=allow_synth,
            )
            status = "complete" if battery.ok else "failed"
            # Per-gate status mirrors battery constituents
            f4_ok = not any(str(f).startswith("F4:") for f in battery.failures) and bool(
                battery.f4.get("computed_sha256")
            ) and not battery.f4.get("failures")
            # Prefer explicit auth dicts
            f4_fail = list(battery.f4.get("failures") or [])
            id_fail = list(battery.identity.get("failures") or [])
            db_fail = list(battery.double_bos.get("failures") or [])
            man_fail = [
                f
                for f in battery.failures
                if str(f).startswith("manifest:")
            ] or list(battery.manifest.get("failures") or [])
            if not battery.ok and not (f4_fail or id_fail or db_fail or man_fail):
                man_fail = list(battery.failures)

            gates["frozen_f4_tolerance_table"] = GateEvidence(
                "frozen_f4_tolerance_table",
                "complete" if battery.ok and not f4_fail else (
                    "failed" if battery.f4.get("computed_sha256") else "blocked"
                ),
                evidence=battery.f4,
                failures=f4_fail
                if f4_fail
                else ([] if battery.ok else ["F4 failed empirical battery"]),
            )
            gates["identities_1_to_9"] = GateEvidence(
                "identities_1_to_9",
                "complete" if battery.ok and not id_fail else (
                    "failed" if battery.identity.get("computed_sha256") else "blocked"
                ),
                evidence=battery.identity,
                failures=id_fail
                if id_fail
                else ([] if battery.ok else ["identity failed empirical battery"]),
            )
            gates["descriptive_double_bos_64_pair_scan"] = GateEvidence(
                "descriptive_double_bos_64_pair_scan",
                "complete" if battery.ok and not db_fail else (
                    "failed" if battery.double_bos.get("computed_sha256") else "blocked"
                ),
                evidence=battery.double_bos,
                failures=db_fail
                if db_fail
                else ([] if battery.ok else ["double-BOS failed empirical battery"]),
            )
            gates["empirical_scientific_battery"] = GateEvidence(
                "empirical_scientific_battery",
                status,
                evidence=battery.to_dict(),
                failures=list(battery.failures),
            )
            _ = f4_ok  # documented: battery consolidates
    else:
        gates["frozen_f4_tolerance_table"] = gate_frozen_f4_tolerance_table(
            f4_table,
            require_empirical_scientific=False,
            expected_model_units=expected_model_units,
            expected_dataset_sha=expected_dataset_sha,
            repo_root=repo_root,
            frozen_prompt_ids=frozen_prompt_ids,
        )
        gates["identities_1_to_9"] = gate_identities_1_to_9(
            identity_suite,
            require_suite=not require_empirical_scientific,
            require_empirical_scientific=False,
            expected_model_units=expected_model_units,
            expected_dataset_sha=expected_dataset_sha,
            repo_root=repo_root,
            frozen_prompt_ids=frozen_prompt_ids,
            f4_artifact=f4_ref,
        )
        gates["descriptive_double_bos_64_pair_scan"] = (
            gate_descriptive_double_bos_64_pair_scan(
                completed=double_bos_scan_completed,
                n_pairs_recorded=double_bos_n_pairs,
                evidence=db_ref,
                require_empirical_scientific=False,
                expected_model_units=expected_model_units,
                expected_dataset_sha=expected_dataset_sha,
                repo_root=repo_root,
                frozen_prompt_ids=frozen_prompt_ids,
            )
        )

    if require_confirmatory:
        if confirmatory_rates is None or confirmatory_n_eligible is None or confirmatory_n_frozen is None:
            gates["per_model_confirmatory_validity"] = GateEvidence(
                "per_model_confirmatory_validity",
                "blocked",
                failures=["confirmatory validity evidence missing"],
            )
        else:
            gates["per_model_confirmatory_validity"] = gate_confirmatory_model_validity(
                trigger_asr=float(confirmatory_rates["trigger_asr"]),
                clean_false_fire=float(confirmatory_rates["clean_false_fire"]),
                first_token_validity=float(confirmatory_rates["first_token_validity"]),
                n_eligible=int(confirmatory_n_eligible),
                n_frozen=int(confirmatory_n_frozen),
            )

    blocked_reasons: list[str] = []
    for g in gates.values():
        if g.status in {"blocked", "failed"}:
            blocked_reasons.append(f"{g.gate_id}:{g.status}")
            blocked_reasons.extend(g.failures)

    return IntegrityGateReport(
        gates=gates,
        progression_blocked=len(blocked_reasons) > 0,
        blocked_reasons=blocked_reasons,
    )
