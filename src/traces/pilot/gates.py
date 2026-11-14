"""Scientific stop / feasibility gates for the Stage 1 pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence  # Any used by native D payloads


# Pre-registered thresholds (v2 / v2.1)
RELIABILITY_JACCARD_K = 10
RELIABILITY_JACCARD_MIN = 0.3
NATIVE_VALIDATION_D_MIN = 0.05


@dataclass
class GateResult:
    name: str
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_discovery_reliability_gate(
    reliability_by_seed: dict[str, Any],
    *,
    k: int = RELIABILITY_JACCARD_K,
    min_jaccard: float = RELIABILITY_JACCARD_MIN,
) -> GateResult:
    """Fail only if *every* seed fails top-k Jaccard reliability."""
    seed_js: dict[str, float] = {}
    for seed, rel in reliability_by_seed.items():
        j = None
        for ov in rel.get("overlaps", []):
            if int(ov.get("k", -1)) == k or float(ov.get("k", -1)) == float(k):
                j = float(ov["jaccard"])
                break
        if j is None and rel.get("overlaps"):
            # fall back to first overlap entry if k-specific missing
            j = float(rel["overlaps"][0]["jaccard"])
        seed_js[str(seed)] = float(j) if j is not None else 0.0

    if not seed_js:
        return GateResult(
            name="discovery_fold_reliability",
            passed=False,
            reason="No reliability records; refusing to continue to held-out.",
            details={"seed_jaccard": seed_js, "threshold": min_jaccard, "k": k},
        )

    all_fail = all(v < min_jaccard for v in seed_js.values())
    if all_fail:
        return GateResult(
            name="discovery_fold_reliability",
            passed=False,
            reason=(
                f"Every seed has top-{k} discovery-fold Jaccard < {min_jaccard}; "
                "stop before held-out evaluation."
            ),
            details={"seed_jaccard": seed_js, "threshold": min_jaccard, "k": k},
        )
    return GateResult(
        name="discovery_fold_reliability",
        passed=True,
        reason=(
            f"At least one seed meets top-{k} Jaccard >= {min_jaccard}; "
            "held-out evaluation permitted."
        ),
        details={"seed_jaccard": seed_js, "threshold": min_jaccard, "k": k},
    )


def _native_abs_max(value: Any) -> float | None:
    """Order-independent max |D| across k budgets or a single scalar."""
    if value is None:
        return None
    if isinstance(value, dict):
        nums = []
        for v in value.values():
            if v is None:
                continue
            if isinstance(v, dict):
                # nested {k: D}
                for vv in v.values():
                    if vv is not None:
                        nums.append(abs(float(vv)))
            else:
                nums.append(abs(float(v)))
        return max(nums) if nums else None
    return abs(float(value))


def evaluate_native_validation_gate(
    native_D_by_recipient: dict[str, Any],
    *,
    min_abs_d: float = NATIVE_VALIDATION_D_MIN,
    k_budgets: Sequence[int] = (5, 10),
) -> GateResult:
    """Native gate: pass if ANY k in k_budgets has |D_native| >= threshold.

    Accepts either ``{recipient: D}`` or ``{recipient: {k: D, ...}}``.
    Aggregation is max_|D| over k — independent of scoring loop order.
    """
    vals = {str(k): v for k, v in native_D_by_recipient.items()}
    per_recipient_max: dict[str, float | None] = {
        rid: _native_abs_max(v) for rid, v in vals.items()
    }
    usable = {
        rid: m
        for rid, m in per_recipient_max.items()
        if m is not None and m >= min_abs_d
    }
    details = {
        "native_D_by_recipient": vals,
        "per_recipient_abs_max_over_k": per_recipient_max,
        "min_abs_d": min_abs_d,
        "k_budgets": list(k_budgets),
        "policy": (
            "Pass if any recipient has max_|D_native(k)| >= min_abs_d for k in k_budgets; "
            "independent of loop order."
        ),
    }
    if not usable:
        return GateResult(
            name="recipient_native_validation_effect",
            passed=False,
            reason=(
                f"No recipient-native validation max_|D| over k{list(k_budgets)} "
                f">= {min_abs_d}; transfer effects must not be interpreted as meaningful "
                "against a null native baseline."
            ),
            details=details,
        )
    return GateResult(
        name="recipient_native_validation_effect",
        passed=True,
        reason=(
            f"At least one recipient has max_|D_native| over k{list(k_budgets)} "
            f">= {min_abs_d}."
        ),
        details={
            **details,
            "usable_recipients": sorted(usable),
        },
    )


def combine_gates(gates: Sequence[GateResult]) -> dict[str, Any]:
    return {
        "gates": [g.to_dict() for g in gates],
        "all_passed": all(g.passed for g in gates),
        "any_failed": any(not g.passed for g in gates),
        "failed_names": [g.name for g in gates if not g.passed],
    }
