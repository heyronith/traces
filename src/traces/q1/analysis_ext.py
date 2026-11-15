"""Numerical analysis helpers for route-denominator gates, k*, H2/H3 (R10)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from traces.q1.analysis import (
    H1_MARGINAL,
    H1_SHARE_PARTIAL,
    H1_SHARE_STRONG,
    Interval,
    ROUTE_DENOM_SHARE_GATE,
    bootstrap_ratio_of_means,
    crossfit_topk_selection,
    rank_pairs_signed,
)
from traces.q1.stats_f1_f2 import F1Draws, percentile_linear


def route_denominator_gate(
    e_route: np.ndarray,
    t_vec: np.ndarray,
    draws: F1Draws,
    *,
    threshold: float = ROUTE_DENOM_SHARE_GATE,
) -> dict[str, Any]:
    """90% lower bound of E_route/T ≥ 0.5 required for route-relative claims."""
    iv = bootstrap_ratio_of_means(e_route, t_vec, draws.full_set)
    return {
        "interval_90": iv,
        "ok": iv.lower >= threshold,
        "threshold": threshold,
    }


def h1_bootstrap_shares(
    e_core: np.ndarray,
    e_route: np.ndarray,
    e_h15: np.ndarray,
    e_h16: np.ndarray,
    draws: F1Draws,
) -> dict[str, Interval]:
    share = bootstrap_ratio_of_means(e_core, e_route, draws.full_set)
    # Conditional addition: (E_pair - E_h) / E_route
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
    # For k10 upper we need a separate input; accept optional
    out: dict[str, Any] = {
        "k_star_le_2": bool(k2 is not None and k2 >= 0.5),
        "k_star_gt_10": bool(k10_share_upper is not None and k10_share_upper < 0.5),
        "undetermined_concentration": False,
    }
    if not out["k_star_le_2"] and not out["k_star_gt_10"]:
        out["undetermined_concentration"] = True
    return out


def crossfit_evaluate_topk(
    *,
    head_effects_fold_a: dict[tuple[int, int], float],
    head_effects_fold_b: dict[tuple[int, int], float],
    per_prompt_effects: dict[tuple[int, int], np.ndarray],
    e_route_prompts: np.ndarray,
    eligible_fold_a_idx: Sequence[int],
    eligible_fold_b_idx: Sequence[int],
    k: int,
    draws: F1Draws,
) -> dict[str, Any]:
    """Select top-k on opposite fold; evaluate joint effect share on eval fold."""
    if draws.crossfit_indeterminate or draws.fold_a is None or draws.fold_b is None:
        return {"indeterminate": True, "reason": draws.crossfit_reason}
    sel_from_a = crossfit_topk_selection(head_effects_fold_a, k)
    sel_from_b = crossfit_topk_selection(head_effects_fold_b, k)

    def joint_effect(prompt_idx: Sequence[int], selected: list[tuple[int, int]]) -> np.ndarray:
        # Approximate joint as sum of selected single-head effects on those prompts
        # (exact joint requires condition margins; synthetic driver supplies joint arrays)
        acc = np.zeros(len(prompt_idx), dtype=np.float64)
        for head in selected:
            acc += per_prompt_effects[head][list(prompt_idx)]
        return acc

    # Evaluate A-selection on B prompts and B-selection on A prompts; pool
    # For point estimate use all eligible indices in each fold
    a_idx = list(eligible_fold_a_idx)
    b_idx = list(eligible_fold_b_idx)
    eff_b = joint_effect(b_idx, sel_from_a)
    eff_a = joint_effect(a_idx, sel_from_b)
    route_b = e_route_prompts[b_idx]
    route_a = e_route_prompts[a_idx]
    pooled_eff = np.concatenate([eff_a, eff_b])
    pooled_route = np.concatenate([route_a, route_b])
    point = float(np.mean(pooled_eff) / np.mean(pooled_route)) if np.mean(pooled_route) else float("nan")
    return {
        "indeterminate": False,
        "k": k,
        "selection_from_a": sel_from_a,
        "selection_from_b": sel_from_b,
        "point_share": point,
    }


def h2_component_coverage(
    component_share_intervals: dict[Any, Interval],
    *,
    upper_threshold: float = 0.25,
) -> dict[str, Any]:
    """H2 requires every component upper90 < 0.25."""
    violators = [
        k for k, iv in component_share_intervals.items() if iv.upper >= upper_threshold
    ]
    return {
        "all_upper_below": len(violators) == 0,
        "violators": violators[:20],
        "n_components": len(component_share_intervals),
    }


def h3_top2_conservation(
    top2_by_seed: dict[Any, frozenset[tuple[int, int]]],
) -> dict[str, Any]:
    """H3a: identical unordered top-2 in ≥4 of 5 seeds; H3b: varies."""
    sets = list(top2_by_seed.values())
    if not sets:
        return {"H3a_count": 0, "varies": False}
    from collections import Counter

    c = Counter(sets)
    best_set, best_count = c.most_common(1)[0]
    return {
        "H3a_count": best_count,
        "H3a_set": best_set,
        "varies": len(c) > 1,
        "n_seeds": len(sets),
    }
