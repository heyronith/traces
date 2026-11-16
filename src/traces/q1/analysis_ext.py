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


def h3_top2_conservation(
    top2_by_seed: dict[Any, frozenset[tuple[int, int]]],
    *,
    require_five_seeds: bool = True,
) -> dict[str, Any]:
    """H3a/H3b only when exactly five distinct primary seeds with valid top-2 sets."""
    seed_check = validate_five_primary_seeds(top2_by_seed, expected_n=5)
    if require_five_seeds and not seed_check["ok"]:
        return {
            "H3a_count": 0,
            "varies": False,
            "n_seeds": seed_check["n_seeds"],
            "indeterminate": True,
            "H3a_eligible": False,
            "H3b_eligible": False,
            "reason": "; ".join(seed_check["failures"]),
            "seed_validation": seed_check,
        }
    sets = list(top2_by_seed.values())
    c = Counter(sets)
    best_set, best_count = c.most_common(1)[0]
    return {
        "H3a_count": best_count,
        "H3a_set": best_set,
        "varies": len(c) > 1,
        "n_seeds": len(sets),
        "indeterminate": False,
        "H3a_eligible": True,
        "H3b_eligible": True,
        "reason": None,
        "seed_validation": seed_check,
    }


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
