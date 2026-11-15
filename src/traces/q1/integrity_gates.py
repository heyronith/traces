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


def gate_identities_1_to_9(
    suite: IdentitySuiteReport | None = None,
    *,
    require_suite: bool = True,
) -> GateEvidence:
    """Identities 1–9. Synthetic suite acceptable for engineering; live model suite
    remains authorization-dependent."""
    if suite is None:
        if require_suite:
            suite = run_synthetic_identity_suite()
        else:
            return GateEvidence(
                "identities_1_to_9",
                "blocked",
                failures=["identity suite evidence missing"],
            )
    return GateEvidence(
        "identities_1_to_9",
        "complete" if suite.all_passed else "failed",
        evidence=suite.to_dict(),
        failures=[] if suite.all_passed else ["one or more identities 1–9 failed"],
    )


def gate_frozen_f4_tolerance_table(
    table: F4CalibrationTable | Mapping[str, Any] | None,
) -> GateEvidence:
    """Frozen F4 tolerance table must be locked with evidence. Empirical lock
    without authorized repeats is BLOCKED — never fabricate."""
    if table is None:
        return GateEvidence(
            "frozen_f4_tolerance_table",
            "blocked",
            evidence={"note": "F4 empirical lock requires authorized model repeats"},
            failures=["F4 tolerance table evidence unavailable"],
        )
    if isinstance(table, F4CalibrationTable):
        locked = bool(table.locked)
        payload = table.to_dict() if hasattr(table, "to_dict") else asdict(table)
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
    completed: bool,
    n_pairs_recorded: int | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> GateEvidence:
    expected = len(enumerate_double_bos_descriptive_pairs())
    if not completed:
        return GateEvidence(
            "descriptive_double_bos_64_pair_scan",
            "blocked",
            evidence={"expected_pairs": expected, **(dict(evidence or {}))},
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
    identity_suite: IdentitySuiteReport | None = None,
    f4_table: F4CalibrationTable | Mapping[str, Any] | None = None,
    double_bos_scan_completed: bool = False,
    double_bos_n_pairs: int | None = None,
    confirmatory_rates: Mapping[str, float] | None = None,
    confirmatory_n_eligible: int | None = None,
    confirmatory_n_frozen: int | None = None,
    require_confirmatory: bool = False,
) -> IntegrityGateReport:
    """Run the B2 gate battery. Missing evidence → blocked progression."""
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

    gates["identities_1_to_9"] = gate_identities_1_to_9(identity_suite)
    gates["frozen_f4_tolerance_table"] = gate_frozen_f4_tolerance_table(f4_table)
    gates["descriptive_double_bos_64_pair_scan"] = gate_descriptive_double_bos_64_pair_scan(
        completed=double_bos_scan_completed, n_pairs_recorded=double_bos_n_pairs
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
