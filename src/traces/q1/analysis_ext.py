"""Numerical analysis helpers for route-denominator gates, k*, H2/H3 (R10).

F2 cross-fitting evaluates **actual jointly patched** top-k head-set effects
(supplied by the execution layer), never sums of individual-head effects.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from traces.q1.analysis import (
    Interval,
    ROUTE_DENOM_SHARE_GATE,
    bootstrap_ratio_of_means,
    crossfit_topk_selection,
)
from traces.q1.component_sets import (
    EXPECTED_L15_L16_PAIRS,
    EXPECTED_UPSTREAM_HEADS,
    validate_effect_vector,
    validate_five_primary_seeds,
)
from traces.q1.stats_f1_f2 import F1Draws, percentile_linear, ratio_of_paired_means


def route_denominator_gate(
    e_route: np.ndarray,
    t_vec: np.ndarray,
    draws: F1Draws,
    *,
    threshold: float = ROUTE_DENOM_SHARE_GATE,
) -> dict[str, Any]:
    """90% lower bound of E_route/T ≥ 0.5 required for route-relative claims."""
    fails = []
    fails.extend(validate_effect_vector(e_route, n_eligible=len(e_route), name="e_route"))
    fails.extend(validate_effect_vector(t_vec, n_eligible=len(t_vec), name="t_vec"))
    if len(e_route) != len(t_vec):
        fails.append("e_route/t_vec length mismatch")
    if fails or len(e_route) == 0:
        return {
            "interval_90": Interval(float("nan"), float("nan"), float("nan")),
            "ok": False,
            "threshold": threshold,
            "indeterminate": True,
            "reason": "; ".join(fails) if fails else "empty vectors",
        }
    iv = bootstrap_ratio_of_means(e_route, t_vec, draws.full_set)
    return {
        "interval_90": iv,
        "ok": bool(np.isfinite(iv.lower) and iv.lower >= threshold),
        "threshold": threshold,
        "indeterminate": not np.isfinite(iv.lower),
        "reason": None if np.isfinite(iv.lower) else "nonfinite route-denom lower bound",
    }


def h1_bootstrap_shares(
    e_core: np.ndarray,
    e_route: np.ndarray,
    e_h15: np.ndarray,
    e_h16: np.ndarray,
    draws: F1Draws,
) -> dict[str, Interval]:
    share = bootstrap_ratio_of_means(e_core, e_route, draws.full_set)
    marg15 = bootstrap_ratio_of_means(e_core - e_h15, e_route, draws.full_set)
    marg16 = bootstrap_ratio_of_means(e_core - e_h16, e_route, draws.full_set)
    return {"share": share, "marginal_h15": marg15, "marginal_h16": marg16}


def classify_k_star(
    k_to_share_lower_90: dict[int, float],
    *,
    k2_share_lower: float | None = None,
    k10_share_upper: float | None = None,
    require_complete_k_scan: bool = True,
    k_scan_complete: bool = True,
) -> dict[str, Any]:
    """k*≤2 if lower90(k=2)≥0.5; k*>10 if upper90(k=10)<0.5; else undetermined.

    Missing required k values or incomplete scans cannot yield definitive
    concentration conclusions.
    """
    k2 = k2_share_lower if k2_share_lower is not None else k_to_share_lower_90.get(2)
    out: dict[str, Any] = {
        "k_star_le_2": False,
        "k_star_gt_10": False,
        "undetermined_concentration": True,
        "k2_share_lower_90": k2,
        "k10_share_upper_90": k10_share_upper,
        "definitive_refused": False,
        "reason": None,
    }
    if require_complete_k_scan and not k_scan_complete:
        out["definitive_refused"] = True
        out["reason"] = "incomplete k=1..10 joint-effect scan; refusing definitive k*"
        return out
    if k2 is None or (k10_share_upper is None and k2 is not None and k2 < 0.5):
        # Cannot claim k*>10 without k10 upper; cannot claim k*≤2 without k2 lower
        if k2 is None and k10_share_upper is None:
            out["definitive_refused"] = True
            out["reason"] = "missing k=2 lower and/or k=10 upper bounds"
            return out
    if k2 is not None and np.isfinite(k2) and k2 >= 0.5:
        out["k_star_le_2"] = True
        out["undetermined_concentration"] = False
    if k10_share_upper is not None and np.isfinite(k10_share_upper) and k10_share_upper < 0.5:
        out["k_star_gt_10"] = True
        out["undetermined_concentration"] = False
    if not out["k_star_le_2"] and not out["k_star_gt_10"]:
        out["undetermined_concentration"] = True
    return out


def selection_key(heads: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Canonical sorted-tuple key for a jointly patched head set."""
    return tuple(sorted((int(L), int(h)) for L, h in heads))


def lookup_joint_effect(
    joint_effects: Mapping[tuple[tuple[int, int], ...], np.ndarray] | None,
    heads: Sequence[tuple[int, int]],
) -> np.ndarray | None:
    if joint_effects is None:
        return None
    key = selection_key(heads)
    if key in joint_effects:
        return np.asarray(joint_effects[key], dtype=np.float64)
    return None


def validate_fold_eval_joint_effects(
    arr: Any,
    *,
    n_eligible: int,
    eval_idx: Sequence[int],
    name: str,
) -> list[str]:
    """Fold-aware joint validation (B1-C10).

    Requires finite measured effects at every opposite-fold evaluation index.
    Allows NaN / unmeasured positions outside the evaluation fold.
    Never treats missing eval measurements as zeros.
    """
    failures: list[str] = []
    try:
        a = np.asarray(arr, dtype=np.float64)
    except (TypeError, ValueError) as e:
        return [f"{name}: not array-convertible ({e})"]
    if a.ndim != 1:
        failures.append(f"{name}: expected 1-d, got shape {a.shape}")
        return failures
    if len(a) != int(n_eligible):
        failures.append(f"{name}: length {len(a)} != n_eligible {n_eligible}")
        return failures
    idx = [int(i) for i in eval_idx]
    if not idx:
        failures.append(f"{name}: empty evaluation fold indices")
        return failures
    if any(i < 0 or i >= len(a) for i in idx):
        failures.append(f"{name}: eval index out of range")
        return failures
    measured = a[np.asarray(idx, dtype=int)]
    if not np.all(np.isfinite(measured)):
        n_bad = int(np.sum(~np.isfinite(measured)))
        failures.append(
            f"{name}: {n_bad}/{len(idx)} required opposite-fold evaluation "
            "positions are nonfinite — refusing zero substitution"
        )
    return failures


def crossfit_evaluate_topk(
    *,
    head_effects_fold_a: dict[tuple[int, int], float],
    head_effects_fold_b: dict[tuple[int, int], float],
    joint_effects: Mapping[tuple[tuple[int, int], ...], np.ndarray],
    e_route_prompts: np.ndarray,
    eligible_fold_a_idx: Sequence[int],
    eligible_fold_b_idx: Sequence[int],
    k: int,
    draws: F1Draws,
    n_eligible: int | None = None,
) -> dict[str, Any]:
    """Select top-k on opposite fold; evaluate **joint** set effect share on eval fold.

    Bootstrap holds selections fixed and resamples within each evaluation fold.
    Invalid bootstrap denominators are recorded as nonfinite replicates without
    silently dropping them from the inference sample size accounting.

    Joint arrays may contain unmeasured (NaN) entries outside the evaluation
    fold; fold-aware validation (B1-C10) requires finite values only at the
    opposite-fold evaluation positions.
    """
    if draws.crossfit_indeterminate or draws.fold_a is None or draws.fold_b is None:
        return {
            "indeterminate": True,
            "reason": draws.crossfit_reason or "empty fold / no fold draws",
            "k": k,
        }
    a_idx = list(eligible_fold_a_idx)
    b_idx = list(eligible_fold_b_idx)
    if not a_idx or not b_idx:
        return {
            "indeterminate": True,
            "reason": "empty eligible fold index lists",
            "k": k,
        }

    n_el = int(n_eligible if n_eligible is not None else len(e_route_prompts))
    route = np.asarray(e_route_prompts, dtype=np.float64)
    route_fails = validate_effect_vector(route, n_eligible=n_el, name="e_route")
    if route_fails:
        return {"indeterminate": True, "reason": "; ".join(route_fails), "k": k}

    sel_from_a = crossfit_topk_selection(head_effects_fold_a, k)
    sel_from_b = crossfit_topk_selection(head_effects_fold_b, k)
    joint_sel_a = lookup_joint_effect(joint_effects, sel_from_a)
    joint_sel_b = lookup_joint_effect(joint_effects, sel_from_b)
    if joint_sel_a is None or joint_sel_b is None:
        return {
            "indeterminate": True,
            "reason": (
                "missing jointly patched effect arrays for selected top-k sets "
                "(refusing individual-head sum approximation)"
            ),
            "k": k,
            "selection_from_a": sel_from_a,
            "selection_from_b": sel_from_b,
        }

    # Selection from A is evaluated on fold B; selection from B on fold A
    fails_a = validate_fold_eval_joint_effects(
        joint_sel_a, n_eligible=n_el, eval_idx=b_idx, name="joint_sel_a(eval=B)"
    )
    fails_b = validate_fold_eval_joint_effects(
        joint_sel_b, n_eligible=n_el, eval_idx=a_idx, name="joint_sel_b(eval=A)"
    )
    if fails_a or fails_b:
        return {
            "indeterminate": True,
            "reason": "; ".join(fails_a + fails_b),
            "k": k,
            "selection_from_a": sel_from_a,
            "selection_from_b": sel_from_b,
            "selection_direction_note": (
                "joint_sel_a=selected_on_A_eval_on_B; "
                "joint_sel_b=selected_on_B_eval_on_A"
            ),
        }

    eff_b = joint_sel_a[b_idx]
    eff_a = joint_sel_b[a_idx]
    route_b = route[b_idx]
    route_a = route[a_idx]
    try:
        point = float(
            ratio_of_paired_means(
                np.concatenate([eff_a, eff_b]),
                np.concatenate([route_a, route_b]),
            )
        )
    except RuntimeError:
        return {
            "indeterminate": True,
            "reason": "invalid (zero) pooled route denominator for point share",
            "k": k,
            "selection_from_a": sel_from_a,
            "selection_from_b": sel_from_b,
        }

    n_boot = int(draws.fold_a.shape[0])
    stats = np.full(n_boot, np.nan, dtype=np.float64)
    n_invalid_den = 0
    for i, (row_a, row_b) in enumerate(zip(draws.fold_a, draws.fold_b)):
        ea = joint_sel_b[np.asarray(a_idx)[row_a]]
        eb = joint_sel_a[np.asarray(b_idx)[row_b]]
        ra = route[np.asarray(a_idx)[row_a]]
        rb = route[np.asarray(b_idx)[row_b]]
        pooled_e = np.concatenate([ea, eb])
        pooled_r = np.concatenate([ra, rb])
        den = float(np.mean(pooled_r))
        if den == 0.0 or not np.isfinite(den):
            n_invalid_den += 1
            # keep NaN in stats — do not silently shrink the inference sample
            continue
        stats[i] = float(np.mean(pooled_e) / den)

    finite = stats[np.isfinite(stats)]
    if len(finite) < 1:
        return {
            "indeterminate": True,
            "reason": "all bootstrap replicates had invalid denominators",
            "k": k,
            "selection_from_a": sel_from_a,
            "selection_from_b": sel_from_b,
            "point_share": point,
            "n_bootstrap_draws": n_boot,
            "n_invalid_denominator": n_invalid_den,
            "n_bootstrap_finite": 0,
        }
    # Percentiles over finite replicates only, but report invalid count for audit
    lo, hi = percentile_linear(finite, [5.0, 95.0])
    if n_invalid_den > 0:
        # Incomplete valid bootstrap → refuse definitive interval as sole support
        return {
            "indeterminate": True,
            "reason": (
                f"{n_invalid_den}/{n_boot} bootstrap replicates had invalid "
                "denominators; refusing definitive share interval"
            ),
            "k": k,
            "selection_from_a": sel_from_a,
            "selection_from_b": sel_from_b,
            "point_share": point,
            "partial_interval_90": {
                "lower": float(lo),
                "upper": float(hi),
                "point": point,
            },
            "n_bootstrap_draws": n_boot,
            "n_invalid_denominator": n_invalid_den,
            "n_bootstrap_finite": int(len(finite)),
        }

    iv = Interval(float(lo), float(hi), point)
    return {
        "indeterminate": False,
        "k": k,
        "selection_from_a": sel_from_a,
        "selection_from_b": sel_from_b,
        "point_share": point,
        "share_interval_90": iv,
        "n_bootstrap_draws": n_boot,
        "n_invalid_denominator": 0,
        "n_bootstrap_finite": int(len(finite)),
    }


def component_route_share_intervals(
    component_effects: Mapping[Any, np.ndarray],
    e_route: np.ndarray,
    draws: F1Draws,
) -> dict[Any, Interval]:
    out: dict[Any, Interval] = {}
    for key, eff in component_effects.items():
        out[key] = bootstrap_ratio_of_means(
            np.asarray(eff, dtype=np.float64),
            e_route,
            draws.full_set,
        )
    return out


def h2_component_coverage(
    component_share_intervals: dict[Any, Interval],
    *,
    upper_threshold: float = 0.25,
    expected_head_keys: frozenset[tuple[int, int]] | None = None,
    expected_pair_keys: frozenset[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """H2 requires every expected component's finite upper90 < 0.25.

    Missing components or nonfinite bounds do **not** satisfy the condition.
    """
    exp_heads = expected_head_keys if expected_head_keys is not None else EXPECTED_UPSTREAM_HEADS
    exp_pairs = expected_pair_keys if expected_pair_keys is not None else EXPECTED_L15_L16_PAIRS

    head_keys_present = {
        k[1:] for k in component_share_intervals if isinstance(k, tuple) and k and k[0] == "head"
    }
    pair_keys_present = {
        k[1:] for k in component_share_intervals if isinstance(k, tuple) and k and k[0] == "pair"
    }
    # Also accept bare (layer,head) / (h15,h16) keys when tagged via parallel maps
    if not head_keys_present and not pair_keys_present:
        # Fallback: keys are already ("head", L, h) or we check lengths only via tagged keys
        pass

    missing_heads = exp_heads - head_keys_present
    missing_pairs = exp_pairs - pair_keys_present
    failures: list[str] = []
    if missing_heads:
        failures.append(f"missing {len(missing_heads)} upstream head components")
    if missing_pairs:
        failures.append(f"missing {len(missing_pairs)} pair components")
    if len(component_share_intervals) == 0:
        failures.append("no component intervals supplied")

    violators = []
    nonfinite = []
    for k, iv in component_share_intervals.items():
        if not np.isfinite(iv.upper):
            nonfinite.append(k)
            continue
        if iv.upper >= upper_threshold:
            violators.append(k)

    if nonfinite:
        failures.append(f"{len(nonfinite)} components have nonfinite upper bounds")

    complete = len(failures) == 0
    all_upper_below = complete and len(violators) == 0 and len(nonfinite) == 0
    return {
        "all_upper_below": all_upper_below,
        "violators": violators[:20],
        "nonfinite": nonfinite[:20],
        "n_components": len(component_share_intervals),
        "n_heads_present": len(head_keys_present),
        "n_pairs_present": len(pair_keys_present),
        "expected_heads": 152,
        "expected_pairs": 64,
        "indeterminate": not complete or len(nonfinite) > 0,
        "reason": "; ".join(failures) if failures else None,
        "missing_head_count": len(missing_heads),
        "missing_pair_count": len(missing_pairs),
    }


def _k_star_le_2_ok(k_star: Mapping[str, Any] | None) -> bool:
    if not isinstance(k_star, Mapping):
        return False
    return bool(k_star.get("k_star_le_2")) and not k_star.get("definitive_refused")


def _k_star_band(k_star: Mapping[str, Any] | None) -> str:
    """Classify frozen k* evidence: le_2 | gt_10 | undetermined | incomplete | missing."""
    if not isinstance(k_star, Mapping):
        return "missing"
    if k_star.get("definitive_refused"):
        return "incomplete"
    if k_star.get("k_star_le_2"):
        return "le_2"
    if k_star.get("k_star_gt_10"):
        return "gt_10"
    return "undetermined"


def _safe_top2_from_effects(
    head_effects: Mapping[tuple[int, int], np.ndarray] | None,
    *,
    n_eligible: int,
) -> tuple[frozenset[tuple[int, int]] | None, dict[str, Any]]:
    """Compute unordered top-2 from a complete upstream signed ranking, or fail."""
    if head_effects is None:
        return None, {"ok": False, "failures": ["head effects missing"]}
    from traces.q1.component_sets import validate_complete_signed_ranking

    chk = validate_complete_signed_ranking(head_effects, n_eligible=n_eligible)
    if not chk["ok"]:
        return None, chk
    # Refuse downstream contamination even if caller passed extras somehow
    upstream_only = {
        k: v for k, v in head_effects.items() if k in EXPECTED_UPSTREAM_HEADS
    }
    if len(upstream_only) != 152:
        return None, {
            "ok": False,
            "failures": ["upstream map must be exactly 152 heads after filtering"],
        }
    try:
        top2 = top2_heads_from_mean_effects(
            upstream_only,
            require_complete_upstream=True,
            n_eligible=n_eligible,
        )
    except ValueError as e:
        return None, {"ok": False, "failures": [str(e)]}
    return top2, {"ok": True, "failures": []}


FROZEN_H3_ROUTE_CLASSES = frozenset(
    {"R-moved", "R-and", "R-or", "R-tag", "R-intermediate", "Indeterminate"}
)


def assess_unit_h3_direction_support(
    *,
    route_class: str,
    h1_full_state: str,
    denom_fwd_ok: bool,
    denom_rev_ok: bool,
    head_effects_fwd: Mapping[tuple[int, int], np.ndarray] | None,
    head_effects_rev: Mapping[tuple[int, int], np.ndarray] | None,
    crossfit_fwd: Mapping[str, Any] | None,
    crossfit_rev: Mapping[str, Any] | None,
    n_eligible: int,
    gates_passed: bool = True,
) -> dict[str, Any]:
    """Per-seed applicable-direction H3 evidence + H3 endpoint status (H3-B).

    Endpoint statuses (owner-approved):
      full_qualifying | h3_indeterminate | determinate_non_support | gate_failed

    Decisive-over-uncertain (owner 2026-11-16): fully observed definitive
    disqualifiers (complete R-moved top-2 disagreement; definitive k*>10) take
    precedence over independent H1-indeterminate or undetermined k* for the
    **H3 endpoint only**; uncertain evidence is still disclosed. Unknown route
    labels raise ``ValueError`` (technical fail-closed), never scientific
    determinate non-support.
    """
    top2_fwd, fwd_rank = _safe_top2_from_effects(
        head_effects_fwd, n_eligible=n_eligible
    )
    top2_rev, rev_rank = _safe_top2_from_effects(
        head_effects_rev, n_eligible=n_eligible
    )
    ks_fwd = (crossfit_fwd or {}).get("k_star")
    ks_rev = (crossfit_rev or {}).get("k_star")
    band_fwd = _k_star_band(ks_fwd if isinstance(ks_fwd, Mapping) else None)
    band_rev = _k_star_band(ks_rev if isinstance(ks_rev, Mapping) else None)
    k_fwd = band_fwd == "le_2"
    k_rev = band_rev == "le_2"

    out: dict[str, Any] = {
        "full_h3_eligible": False,
        "top2_for_h3": None,
        "top2_forward": sorted(top2_fwd) if top2_fwd is not None else None,
        "top2_reverse": sorted(top2_rev) if top2_rev is not None else None,
        "applicable_directions": [],
        "route_class": route_class,
        "h1_state": h1_full_state,
        "gates_passed": gates_passed,
        "denom_fwd_ok": bool(denom_fwd_ok),
        "denom_rev_ok": bool(denom_rev_ok),
        "ranking_fwd_ok": bool(fwd_rank.get("ok")),
        "ranking_rev_ok": bool(rev_rank.get("ok")),
        "k_star_le_2_fwd": k_fwd,
        "k_star_le_2_rev": k_rev,
        "k_star_band_fwd": band_fwd,
        "k_star_band_rev": band_rev,
        "rmoved_top2_agree": None,
        "reason": None,
        "support_class": "non_supporting",
        "h3_endpoint_status": "determinate_non_support",
        "uncertainty_kind": None,
        "preserved_uncertainty": None,
    }

    def _finish(
        *,
        status: str,
        support_class: str,
        reason: str,
        uncertainty_kind: str | None = None,
        directions: list[str] | None = None,
        preserved_uncertainty: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        out["h3_endpoint_status"] = status
        out["support_class"] = support_class
        out["reason"] = reason
        out["uncertainty_kind"] = uncertainty_kind
        if directions is not None:
            out["applicable_directions"] = directions
        if preserved_uncertainty is not None:
            out["preserved_uncertainty"] = dict(preserved_uncertainty)
        out["full_h3_eligible"] = False
        out["top2_for_h3"] = None
        return out

    if not gates_passed:
        return _finish(
            status="gate_failed",
            support_class="gate_failed",
            reason="gate-failed; no full H3 support",
        )

    if route_class not in FROZEN_H3_ROUTE_CLASSES:
        raise ValueError(
            f"unknown route_class {route_class!r} — refuse scientific H3 "
            f"endpoint; frozen vocabulary is {sorted(FROZEN_H3_ROUTE_CLASSES)} "
            "(technical fail-closed; not determinate_non_support)"
        )

    if route_class == "Indeterminate":
        return _finish(
            status="h3_indeterminate",
            support_class="route_indeterminate",
            reason="route_class Indeterminate is H3-indeterminate (§5.4(3))",
            uncertainty_kind="route_indeterminate",
        )

    if route_class in {"R-tag", "R-intermediate"}:
        return _finish(
            status="determinate_non_support",
            support_class="route_inapplicable",
            reason=(
                f"route_class {route_class} determinate non-support / inapplicable "
                "for full H3"
            ),
        )

    if route_class == "R-and":
        directions = ["forward"]
    elif route_class == "R-or":
        directions = ["reverse"]
    else:
        directions = ["forward", "reverse"]
    out["applicable_directions"] = directions

    def _uncertainty_payload() -> dict[str, Any]:
        return {
            "h1_state": h1_full_state,
            "k_star_band_fwd": band_fwd,
            "k_star_band_rev": band_rev,
            "top2_forward": out["top2_forward"],
            "top2_reverse": out["top2_reverse"],
            "rmoved_top2_agree": out["rmoved_top2_agree"],
        }

    # --- Denominator applicability (before decisive / uncertain H3 checks) ---
    if route_class == "R-and" and not denom_fwd_ok:
        return _finish(
            status="determinate_non_support",
            support_class="denom_inapplicable",
            reason="R-and forward route-denominator gate failed (direction inapplicable)",
            directions=directions,
        )
    if route_class == "R-or" and not denom_rev_ok:
        return _finish(
            status="determinate_non_support",
            support_class="denom_inapplicable",
            reason="R-or reverse route-denominator gate failed (direction inapplicable)",
            directions=directions,
        )
    if route_class == "R-moved" and (not denom_fwd_ok or not denom_rev_ok):
        failed = []
        if not denom_fwd_ok:
            failed.append("forward")
        if not denom_rev_ok:
            failed.append("reverse")
        return _finish(
            status="determinate_non_support",
            support_class="denom_inapplicable",
            reason=(
                "R-moved route-denominator failed for "
                + ",".join(failed)
                + " (direction inapplicable)"
            ),
            directions=directions,
        )

    ranking_fwd_ok = bool(fwd_rank.get("ok")) and top2_fwd is not None
    ranking_rev_ok = bool(rev_rank.get("ok")) and top2_rev is not None
    if route_class == "R-moved":
        out["rmoved_top2_agree"] = (
            top2_fwd == top2_rev if (ranking_fwd_ok and ranking_rev_ok) else None
        )

    # --- Decisive-over-uncertain: definitive disqualifiers first ---
    # R-moved: only valid complete BOTH maps can establish top-2 disagreement
    if (
        route_class == "R-moved"
        and ranking_fwd_ok
        and ranking_rev_ok
        and top2_fwd != top2_rev
    ):
        return _finish(
            status="determinate_non_support",
            support_class="directional_partial",
            reason=(
                "R-moved forward/reverse top-2 disagree — determinate non-support "
                "for full H3 (decisive-over-uncertain); uncertain H1/k* preserved"
            ),
            directions=directions,
            preserved_uncertainty=_uncertainty_payload(),
        )

    # Definitive k*>10 in any required direction disqualifies full H3
    required_bands: list[tuple[str, str]] = []
    if route_class == "R-and":
        required_bands = [(band_fwd, "R-and forward")]
    elif route_class == "R-or":
        required_bands = [(band_rev, "R-or reverse")]
    else:
        required_bands = [
            (band_fwd, "R-moved forward"),
            (band_rev, "R-moved reverse"),
        ]
    for band, label in required_bands:
        if band == "gt_10":
            return _finish(
                status="determinate_non_support",
                support_class="k_star_gt_10",
                reason=(
                    f"{label} definitive k*>10 — determinate non-support for full H3 "
                    "(decisive-over-uncertain); other-direction uncertainty preserved"
                ),
                directions=directions,
                preserved_uncertainty=_uncertainty_payload(),
            )

    # Explicit H1-pass is determinate non-support (after decisive H3 checks)
    if h1_full_state == "H1-pass":
        return _finish(
            status="determinate_non_support",
            support_class="h1_not_fail",
            reason="explicit H1-pass — determinate non-support for H3",
            directions=directions,
        )

    # Missing/incomplete required rankings — technical, never invent disagreement
    if route_class == "R-and" and not ranking_fwd_ok:
        return _finish(
            status="h3_indeterminate",
            support_class="incomplete_ranking",
            reason="R-and incomplete forward 152-head ranking",
            uncertainty_kind="technical_missing_ranking",
            directions=directions,
        )
    if route_class == "R-or" and not ranking_rev_ok:
        return _finish(
            status="h3_indeterminate",
            support_class="incomplete_ranking",
            reason="R-or incomplete reverse 152-head ranking",
            uncertainty_kind="technical_missing_ranking",
            directions=directions,
        )
    if route_class == "R-moved" and (not ranking_fwd_ok or not ranking_rev_ok):
        return _finish(
            status="h3_indeterminate",
            support_class="incomplete_ranking",
            reason=(
                "R-moved incomplete forward/reverse 152-head ranking — cannot "
                "claim observed top-2 disagreement without both valid maps"
            ),
            uncertainty_kind="technical_missing_ranking",
            directions=directions,
        )

    # Remaining uncertainty without a definitive disqualifier
    if h1_full_state == "H1-indeterminate":
        return _finish(
            status="h3_indeterminate",
            support_class="h1_not_fail",
            reason="H1-indeterminate on applicable route — H3-indeterminate",
            uncertainty_kind="h1_indeterminate",
            directions=directions,
            preserved_uncertainty=_uncertainty_payload(),
        )
    if h1_full_state != "H1-fail":
        return _finish(
            status="h3_indeterminate",
            support_class="h1_not_fail",
            reason=f"unrecognized H1 state {h1_full_state!r} — treat as H3-indeterminate",
            uncertainty_kind="h1_unrecognized",
            directions=directions,
        )

    for band, label in required_bands:
        if band == "le_2":
            continue
        if band == "undetermined":
            return _finish(
                status="h3_indeterminate",
                support_class="k_star_undetermined",
                reason=f"{label} k* in undetermined concentration band",
                uncertainty_kind="k_star_undetermined",
                directions=directions,
                preserved_uncertainty=_uncertainty_payload(),
            )
        return _finish(
            status="h3_indeterminate",
            support_class="k_star_incomplete",
            reason=f"{label} k* incomplete/missing — technical H3-indeterminate",
            uncertainty_kind="technical_missing_k_star",
            directions=directions,
            preserved_uncertainty=_uncertainty_payload(),
        )

    # Full qualifying
    if route_class == "R-and":
        out["top2_for_h3"] = top2_fwd
    elif route_class == "R-or":
        out["top2_for_h3"] = top2_rev
    else:
        out["top2_for_h3"] = top2_fwd  # agrees with reverse
    out["full_h3_eligible"] = True
    out["support_class"] = "full"
    out["h3_endpoint_status"] = "full_qualifying"
    out["reason"] = None
    return out


def _normalize_h3_endpoint_status(rec: Mapping[str, Any]) -> str:
    """Map a seed record to an H3 endpoint status."""
    explicit = rec.get("h3_endpoint_status")
    if explicit in {
        "full_qualifying",
        "h3_indeterminate",
        "determinate_non_support",
        "gate_failed",
        "technical_invalid",
    }:
        return str(explicit)
    if bool(rec.get("full_h3_eligible")):
        return "full_qualifying"
    sc = str(rec.get("support_class") or "")
    if sc == "gate_failed" or rec.get("gates_passed") is False:
        return "gate_failed"
    if sc in {
        "k_star_undetermined",
        "k_star_incomplete",
        "incomplete_ranking",
        "route_indeterminate",
        "technical_invalid",
    }:
        return "h3_indeterminate" if sc != "technical_invalid" else "technical_invalid"
    if sc == "h1_not_fail" and rec.get("h1_state") == "H1-indeterminate":
        return "h3_indeterminate"
    if str(rec.get("route_class")) == "Indeterminate":
        return "h3_indeterminate"
    return "determinate_non_support"


def _validate_h3_seed_record_consistency(
    sid: Any, rec: Mapping[str, Any]
) -> list[str]:
    """Refuse contradictory status/eligibility/top-2 technical records."""
    errs: list[str] = []
    status = rec.get("h3_endpoint_status")
    eligible = bool(rec.get("full_h3_eligible"))
    top2 = rec.get("top2_for_h3")
    if isinstance(top2, list):
        try:
            top2_fs: frozenset[tuple[int, int]] | None = frozenset(
                tuple(x) for x in top2
            )
        except TypeError:
            top2_fs = None
            errs.append(f"{sid}: top2_for_h3 list is not a valid head-pair set")
    elif isinstance(top2, frozenset):
        top2_fs = top2
    elif top2 is None:
        top2_fs = None
    else:
        top2_fs = None
        errs.append(f"{sid}: top2_for_h3 has unsupported type {type(top2)}")

    if status == "full_qualifying" and not eligible:
        errs.append(
            f"{sid}: h3_endpoint_status=full_qualifying with full_h3_eligible=False "
            "(contradictory technical record)"
        )
    if eligible and status not in {None, "full_qualifying"}:
        errs.append(
            f"{sid}: full_h3_eligible=True with non-qualifying status {status!r} "
            "(contradictory technical record)"
        )
    if eligible or status == "full_qualifying":
        if not isinstance(top2_fs, frozenset) or len(top2_fs) != 2:
            errs.append(
                f"{sid}: qualifying/full_qualifying requires exact unordered "
                "frozenset/list of two head pairs"
            )
        elif any(
            not (isinstance(p, tuple) and len(p) == 2) for p in top2_fs
        ):
            errs.append(f"{sid}: malformed head id in top2_for_h3")
    if status == "technical_invalid":
        errs.append(f"{sid}: technical_invalid H3 endpoint record")
    return errs


def h3_study_level_from_seed_records(
    seed_records: Mapping[Any, Mapping[str, Any]],
    *,
    expected_roster: Sequence[Any] | None = None,
    study_inconclusive: bool = False,
    roster_mismatch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Study-level H3a/H3b from per-seed eligibility records (fixed five identities).

    H3-A: H3a and H3b are mutually exclusive under §5.4(4) 4-of-5 conservation.
    H3-B: H3-endpoint INCONCLUSIVE if route study INCONCLUSIVE or
    count(gate_failed + h3_indeterminate) ≥ 2 among the fixed five.
    Contradictory technical records validation-block the study (no false H3 claim).
    """
    if roster_mismatch:
        return {
            "H3a_count": 0,
            "H3a_set": None,
            "varies": False,
            "n_seeds": 5,
            "n_qualifying": 0,
            "indeterminate": True,
            "H3a_eligible": False,
            "H3b_eligible": False,
            "H3a": None,
            "H3b": None,
            "reason": roster_mismatch.get("reason")
            or "frozen primary roster mismatch — canonical H3 refused (H3-C)",
            "non_supporting": {},
            "per_seed": {},
            "study_inconclusive_precedence": False,
            "h3_endpoint_inconclusive": True,
            "h3_endpoint_decision": "INCONCLUSIVE_ROSTER_MISMATCH",
            "qualifying_seeds": [],
            "roster_mismatch": dict(roster_mismatch),
            "endpoint_counts": {},
            "validation_blocked": False,
        }

    roster = list(expected_roster) if expected_roster is not None else list(seed_records.keys())
    failures: list[str] = []
    if len(roster) != 5:
        failures.append(f"expected_roster must have exactly 5 primary identities, got {len(roster)}")
    if len(set(roster)) != len(roster):
        failures.append("duplicate identities in expected_roster")
    for sid in roster:
        if sid not in seed_records:
            failures.append(f"missing seed record for roster identity {sid!r}")

    non_supporting: dict[str, Any] = {}
    qualifying_sets: dict[Any, frozenset[tuple[int, int]]] = {}
    per_seed: dict[str, Any] = {}
    endpoint_counts = {
        "full_qualifying": 0,
        "h3_indeterminate": 0,
        "determinate_non_support": 0,
        "gate_failed": 0,
        "technical_invalid": 0,
    }
    validation_errors: list[str] = []
    for sid in roster:
        rec = dict(seed_records.get(sid) or {})
        validation_errors.extend(_validate_h3_seed_record_consistency(sid, rec))
        status = _normalize_h3_endpoint_status(rec)
        # Never count contradictory full_qualifying+ineligible as qualifying endpoint
        eligible_flag = bool(rec.get("full_h3_eligible"))
        top2 = rec.get("top2_for_h3")
        if isinstance(top2, list):
            try:
                top2 = frozenset(tuple(x) for x in top2)
            except TypeError:
                top2 = None
        consistent_full = (
            status == "full_qualifying"
            and eligible_flag
            and isinstance(top2, frozenset)
            and len(top2) == 2
        )
        if status == "full_qualifying" and not consistent_full:
            # Do not increment full_qualifying count for inconsistent records
            endpoint_counts["technical_invalid"] = (
                endpoint_counts.get("technical_invalid", 0) + 1
            )
        else:
            endpoint_counts[status] = endpoint_counts.get(status, 0) + 1

        if consistent_full:
            qualifying_sets[sid] = top2  # type: ignore[assignment]
            per_seed[str(sid)] = {
                "full_h3_eligible": True,
                "top2": sorted(top2),  # type: ignore[arg-type]
                "reason": None,
                "route_class": rec.get("route_class"),
                "support_class": rec.get("support_class", "full"),
                "h3_endpoint_status": "full_qualifying",
            }
        else:
            reason = rec.get("reason") or (
                "non-supporting / inapplicable for full H3"
            )
            non_supporting[str(sid)] = reason
            per_seed[str(sid)] = {
                "full_h3_eligible": False,
                "top2": sorted(top2) if isinstance(top2, frozenset) else rec.get("top2_for_h3"),
                "top2_forward": rec.get("top2_forward"),
                "top2_reverse": rec.get("top2_reverse"),
                "reason": reason,
                "route_class": rec.get("route_class"),
                "support_class": rec.get("support_class", "non_supporting"),
                "h3_endpoint_status": status
                if status != "full_qualifying"
                else "technical_invalid",
                "uncertainty_kind": rec.get("uncertainty_kind"),
            }

    n_qualifying = len(qualifying_sets)
    n_uncertain = endpoint_counts["gate_failed"] + endpoint_counts["h3_indeterminate"]
    h3_endpoint_inconclusive = bool(study_inconclusive) or (n_uncertain >= 2)

    def _blocked(reason: str, **extra: Any) -> dict[str, Any]:
        return {
            "H3a_count": 0,
            "H3a_set": None,
            "varies": False,
            "n_seeds": len(roster),
            "n_qualifying": 0 if extra.get("validation_blocked") else n_qualifying,
            "indeterminate": True,
            "H3a_eligible": False,
            "H3b_eligible": False,
            "H3a": None,
            "H3b": None,
            "reason": reason,
            "non_supporting": non_supporting,
            "per_seed": per_seed,
            "study_inconclusive_precedence": bool(study_inconclusive),
            "h3_endpoint_inconclusive": True,
            "h3_endpoint_decision": extra.pop(
                "h3_endpoint_decision", "INCONCLUSIVE"
            ),
            "qualifying_seeds": []
            if extra.get("validation_blocked")
            else [str(s) for s in qualifying_sets],
            "endpoint_counts": dict(endpoint_counts),
            "minority_pairs": None,
            **extra,
        }

    if validation_errors:
        return _blocked(
            "H3 validation-blocked: contradictory/invalid seed records — "
            + "; ".join(validation_errors),
            validation_blocked=True,
            h3_endpoint_decision="VALIDATION_BLOCKED",
            validation_errors=list(validation_errors),
            indeterminate=True,
        )

    if failures:
        return _blocked("; ".join(failures), indeterminate=True)

    if h3_endpoint_inconclusive:
        why = []
        if study_inconclusive:
            why.append("route-level study INCONCLUSIVE")
        if n_uncertain >= 2:
            why.append(
                f"H3-endpoint uncertainty: gate_failed="
                f"{endpoint_counts['gate_failed']}, h3_indeterminate="
                f"{endpoint_counts['h3_indeterminate']} (≥2 → §5.4(3))"
            )
        return _blocked(
            "H3 study-level INCONCLUSIVE precedence (§5.4(3)): " + "; ".join(why),
            indeterminate=False,
            study_inconclusive_precedence=bool(study_inconclusive),
        )

    if n_qualifying < 4:
        return {
            "H3a_count": 0,
            "H3a_set": None,
            "varies": False,
            "n_seeds": 5,
            "n_qualifying": n_qualifying,
            "indeterminate": False,
            "H3a_eligible": False,
            "H3b_eligible": False,
            "H3a": None,
            "H3b": None,
            "reason": (
                f"only {n_qualifying} of 5 primary models individually qualify "
                "for full H3 (§5.4(4) requires ≥4)"
            ),
            "non_supporting": non_supporting,
            "per_seed": per_seed,
            "study_inconclusive_precedence": False,
            "h3_endpoint_inconclusive": False,
            "h3_endpoint_decision": "NOT_SUPPORTED",
            "qualifying_seeds": [str(s) for s in qualifying_sets],
            "endpoint_counts": dict(endpoint_counts),
            "minority_pairs": None,
            "validation_blocked": False,
        }

    c = Counter(qualifying_sets.values())
    best_set, best_count = c.most_common(1)[0]
    varies = len(c) > 1
    minority = {
        str(sid): sorted(fs)
        for sid, fs in qualifying_sets.items()
        if fs != best_set
    }
    result: dict[str, Any] = {
        "H3a_count": best_count,
        "H3a_set": best_set,
        "varies": varies,
        "n_seeds": 5,
        "n_qualifying": n_qualifying,
        "indeterminate": False,
        "H3a_eligible": True,
        "H3b_eligible": True,
        "H3a": None,
        "H3b": None,
        "reason": None,
        "non_supporting": non_supporting,
        "per_seed": per_seed,
        "study_inconclusive_precedence": False,
        "h3_endpoint_inconclusive": False,
        "h3_endpoint_decision": None,
        "qualifying_seeds": [str(s) for s in qualifying_sets],
        "qualifying_set_counts": {str(sorted(s)): n for s, n in c.items()},
        "endpoint_counts": dict(endpoint_counts),
        "minority_pairs": minority or None,
        "validation_blocked": False,
    }

    # H3-A: mutually exclusive — conserved 4/5 → H3a only; else varying → H3b only
    if best_count >= 4:
        result["H3a"] = "supported"
        result["H3b"] = None
        result["H3b_eligible"] = False
        result["h3_endpoint_decision"] = "H3a_SUPPORTED"
        if varies:
            result["reason"] = (
                "H3a supported under 4/5 conservation; dissenting qualifying "
                "pair(s) disclosed descriptively — H3b not supported (H3-A)"
            )
    elif varies:
        result["H3b"] = "supported"
        result["H3a"] = None
        result["H3a_eligible"] = False
        result["h3_endpoint_decision"] = "H3b_SUPPORTED"
    else:
        result["H3a_eligible"] = False
        result["H3b_eligible"] = False
        result["h3_endpoint_decision"] = "NOT_SUPPORTED"
        result["reason"] = (
            "≥4 qualify but neither 4/5 conservation nor qualifying variation"
        )
    return result


def h3_top2_conservation(
    top2_by_seed: dict[Any, frozenset[tuple[int, int]] | None],
    *,
    require_five_seeds: bool = True,
    nonqualifying_seeds: Mapping[Any, str] | None = None,
    expected_roster: Sequence[Any] | None = None,
    study_inconclusive: bool = False,
) -> dict[str, Any]:
    """Study-level H3 conservation over a fixed five-seed primary roster.

    Qualifying entries supply frozenset top-2 sets. ``None`` values or keys listed
    in ``nonqualifying_seeds`` count as non-supporting (not missing-data erasure
    of a 4-of-5 claim). Legacy call sites that pass exactly five valid sets keep
    prior conservation counting; partial maps without a roster remain ineligible.
    """
    nonqual = dict(nonqualifying_seeds or {})
    if expected_roster is not None:
        roster = list(expected_roster)
    elif require_five_seeds:
        # Legacy: exactly five present keys required unless Nones mark nonqualifying
        roster = list(top2_by_seed.keys())
        if any(v is None for v in top2_by_seed.values()) or nonqual:
            # Allow 4 qualifying + nonqualifying markers when map has 5 slots
            pass
        else:
            seed_check = validate_five_primary_seeds(
                {k: v for k, v in top2_by_seed.items() if isinstance(v, frozenset)},
                expected_n=5,
            )
            if not seed_check["ok"]:
                return {
                    "H3a_count": 0,
                    "varies": False,
                    "n_seeds": seed_check["n_seeds"],
                    "n_qualifying": 0,
                    "indeterminate": True,
                    "H3a_eligible": False,
                    "H3b_eligible": False,
                    "H3a": None,
                    "H3b": None,
                    "reason": "; ".join(seed_check["failures"]),
                    "seed_validation": seed_check,
                    "non_supporting": {str(k): "absent_from_map" for k in []},
                    "per_seed": {},
                    "study_inconclusive_precedence": False,
                    "qualifying_seeds": [],
                }
    else:
        roster = list(top2_by_seed.keys())

    records: dict[Any, dict[str, Any]] = {}
    for sid in roster:
        if sid in nonqual:
            records[sid] = {
                "full_h3_eligible": False,
                "top2_for_h3": None,
                "reason": nonqual[sid],
                "support_class": "non_supporting",
            }
            continue
        val = top2_by_seed.get(sid)
        if isinstance(val, frozenset) and len(val) == 2:
            records[sid] = {
                "full_h3_eligible": True,
                "top2_for_h3": val,
                "reason": None,
                "support_class": "full",
            }
        else:
            records[sid] = {
                "full_h3_eligible": False,
                "top2_for_h3": val if isinstance(val, frozenset) else None,
                "reason": nonqual.get(sid, "missing or invalid top-2"),
                "support_class": "non_supporting",
            }
    # Include any nonqual keys not already on roster
    for sid, reason in nonqual.items():
        if sid not in records:
            records[sid] = {
                "full_h3_eligible": False,
                "top2_for_h3": None,
                "reason": reason,
                "support_class": "non_supporting",
            }
            if sid not in roster and expected_roster is None and len(roster) < 5:
                roster.append(sid)

    return h3_study_level_from_seed_records(
        records,
        expected_roster=roster if (expected_roster is not None or len(roster) == 5) else roster,
        study_inconclusive=study_inconclusive,
    )


def top2_heads_from_mean_effects(
    head_effects: Mapping[tuple[int, int], np.ndarray],
    prompt_idx: Sequence[int] | None = None,
    *,
    require_complete_upstream: bool = False,
    n_eligible: int | None = None,
) -> frozenset[tuple[int, int]]:
    """Unordered top-2 heads by signed mean effect; tie-break (layer, head)."""
    if require_complete_upstream:
        from traces.q1.component_sets import validate_complete_signed_ranking

        n = n_eligible if n_eligible is not None else len(next(iter(head_effects.values())))
        chk = validate_complete_signed_ranking(head_effects, n_eligible=n)
        if not chk["ok"]:
            raise ValueError(f"incomplete signed ranking: {chk['failures']}")
    means: dict[tuple[int, int], float] = {}
    for key, arr in head_effects.items():
        a = np.asarray(arr, dtype=np.float64)
        if prompt_idx is not None:
            a = a[list(prompt_idx)]
        if not np.all(np.isfinite(a)):
            raise ValueError(f"nonfinite effects for head {key}")
        means[key] = float(np.mean(a))
    ranked = sorted(means.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return frozenset(h for h, _ in ranked[:2])
