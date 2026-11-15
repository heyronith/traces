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
from traces.q1.stats_f1_f2 import F1Draws, percentile_linear, ratio_of_paired_means


def route_denominator_gate(
    e_route: np.ndarray,
    t_vec: np.ndarray,
    draws: F1Draws,
    *,
    threshold: float = ROUTE_DENOM_SHARE_GATE,
) -> dict[str, Any]:
    """90% lower bound of E_route/T ≥ 0.5 required for route-relative claims."""
    if len(e_route) == 0 or len(t_vec) == 0:
        return {
            "interval_90": Interval(float("nan"), float("nan"), float("nan")),
            "ok": False,
            "threshold": threshold,
            "indeterminate": True,
            "reason": "empty route or T vectors",
        }
    iv = bootstrap_ratio_of_means(e_route, t_vec, draws.full_set)
    return {
        "interval_90": iv,
        "ok": iv.lower >= threshold,
        "threshold": threshold,
        "indeterminate": False,
        "reason": None,
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
) -> dict[str, Any]:
    """k*≤2 if lower90(k=2)≥0.5; k*>10 if upper90(k=10)<0.5; else undetermined."""
    k2 = k2_share_lower if k2_share_lower is not None else k_to_share_lower_90.get(2)
    out: dict[str, Any] = {
        "k_star_le_2": bool(k2 is not None and k2 >= 0.5),
        "k_star_gt_10": bool(k10_share_upper is not None and k10_share_upper < 0.5),
        "undetermined_concentration": False,
        "k2_share_lower_90": k2,
        "k10_share_upper_90": k10_share_upper,
    }
    if not out["k_star_le_2"] and not out["k_star_gt_10"]:
        out["undetermined_concentration"] = True
    return out


def selection_key(heads: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Canonical frozenset-as-sorted-tuple key for a jointly patched head set."""
    return tuple(sorted((int(L), int(h)) for L, h in heads))


def lookup_joint_effect(
    joint_effects: Mapping[tuple[tuple[int, int], ...], np.ndarray] | None,
    heads: Sequence[tuple[int, int]],
) -> np.ndarray | None:
    """Return per-prompt joint effect array for the exact head set, or None."""
    if joint_effects is None:
        return None
    key = selection_key(heads)
    if key in joint_effects:
        return np.asarray(joint_effects[key], dtype=np.float64)
    # Also accept unordered key variants already sorted
    return None


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
) -> dict[str, Any]:
    """Select top-k on opposite fold; evaluate **joint** set effect share on eval fold.

    Bootstrap holds the selected head sets fixed and resamples prompts within each
    evaluation fold (F1 fold-specific draws), then pools prompt-level margins.
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

    route = np.asarray(e_route_prompts, dtype=np.float64)
    # Fixed selections: A→eval B, B→eval A; pool prompt-level for point + bootstrap
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

    # Fold-specific bootstrap with fixed selections
    stats: list[float] = []
    for row_a, row_b in zip(draws.fold_a, draws.fold_b):
        # row_* index into fold-local arrays (length n_a / n_b)
        ea = joint_sel_b[np.asarray(a_idx)[row_a]]
        eb = joint_sel_a[np.asarray(b_idx)[row_b]]
        ra = route[np.asarray(a_idx)[row_a]]
        rb = route[np.asarray(b_idx)[row_b]]
        pooled_e = np.concatenate([ea, eb])
        pooled_r = np.concatenate([ra, rb])
        den = float(np.mean(pooled_r))
        if den == 0.0:
            stats.append(float("nan"))
        else:
            stats.append(float(np.mean(pooled_e) / den))
    arr = np.asarray(stats, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    if len(finite) < 1:
        return {
            "indeterminate": True,
            "reason": "all bootstrap replicates had invalid denominators",
            "k": k,
            "selection_from_a": sel_from_a,
            "selection_from_b": sel_from_b,
            "point_share": point,
        }
    lo, hi = percentile_linear(finite, [5.0, 95.0])
    iv = Interval(float(lo), float(hi), point)
    return {
        "indeterminate": False,
        "k": k,
        "selection_from_a": sel_from_a,
        "selection_from_b": sel_from_b,
        "point_share": point,
        "share_interval_90": iv,
        "n_bootstrap_finite": int(len(finite)),
    }


def component_route_share_intervals(
    component_effects: Mapping[Any, np.ndarray],
    e_route: np.ndarray,
    draws: F1Draws,
) -> dict[Any, Interval]:
    """Bootstrap 90% intervals for every component's route-effect share."""
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
) -> dict[str, Any]:
    """H2 requires every component upper90 < 0.25; overlap → not all_upper_below."""
    if not component_share_intervals:
        return {
            "all_upper_below": False,
            "violators": [],
            "n_components": 0,
            "indeterminate": True,
            "reason": "no component intervals supplied",
        }
    violators = [
        k for k, iv in component_share_intervals.items() if iv.upper >= upper_threshold
    ]
    return {
        "all_upper_below": len(violators) == 0,
        "violators": violators[:20],
        "n_components": len(component_share_intervals),
        "indeterminate": False,
        "reason": None,
    }


def h3_top2_conservation(
    top2_by_seed: dict[Any, frozenset[tuple[int, int]]],
) -> dict[str, Any]:
    """H3a: identical unordered top-2 in ≥4 of 5 seeds; H3b: varies."""
    sets = list(top2_by_seed.values())
    if not sets:
        return {
            "H3a_count": 0,
            "varies": False,
            "n_seeds": 0,
            "indeterminate": True,
            "reason": "no top-2 sets supplied",
        }
    c = Counter(sets)
    best_set, best_count = c.most_common(1)[0]
    return {
        "H3a_count": best_count,
        "H3a_set": best_set,
        "varies": len(c) > 1,
        "n_seeds": len(sets),
        "indeterminate": False,
        "reason": None,
    }


def top2_heads_from_mean_effects(
    head_effects: Mapping[tuple[int, int], np.ndarray],
    prompt_idx: Sequence[int] | None = None,
) -> frozenset[tuple[int, int]]:
    """Unordered top-2 heads by signed mean effect; tie-break (layer, head)."""
    means: dict[tuple[int, int], float] = {}
    for key, arr in head_effects.items():
        a = np.asarray(arr, dtype=np.float64)
        if prompt_idx is not None:
            a = a[list(prompt_idx)]
        means[key] = float(np.mean(a))
    ranked = sorted(means.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    return frozenset(h for h, _ in ranked[:2])
