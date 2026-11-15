"""Frozen §§4–7 scientific analysis pipeline (CPU; no inference).

Operates on precomputed per-prompt margins and head/pair effects. Thresholds and
decision rules are taken from protocol v1.3 and must not be altered here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

from traces.q1.stats_f1_f2 import (
    F1Draws,
    assign_f2_folds,
    draw_f1_resamples,
    percentile_linear,
    percentile_rank_inverted_cdf,
    ratio_of_paired_means,
)

THETA_HI = 0.5
THETA_LO = 0.2
T_GATE_NAT = 1.0
ROUTE_DENOM_SHARE_GATE = 0.5  # 90% lower bound ≥ 0.5 for route-relative claims
H1_RANK_UPPER = 3
H1_SHARE_STRONG = 0.5
H1_SHARE_PARTIAL = 0.25
H1_MARGINAL = 0.1
ELIGIBILITY_MIN_FRAC = 0.90
ASR_MIN = 0.90
FALSE_FIRE_MAX = 0.05
FIRST_TOKEN_MIN = 0.90
DEV_INELIGIBLE_MAX = 0.10
EXPLORATORY_STOP_RANK = 10

RouteClass = Literal[
    "R-moved",
    "R-and",
    "R-or",
    "R-tag",
    "R-intermediate",
    "Indeterminate",
]
# Decision states for evaluate_h1_direction (exactly three + inapplicable).
# ``unresolved_marginal_contribution`` is an annotation flag, not a state.
H1State = Literal[
    "H1-pass",
    "H1-fail",
    "H1-indeterminate",
    "inapplicable",
]

# Canonical labels that count toward study-level inconclusive (≥2 of 5)
STUDY_INCONCLUSIVE_LABELS = frozenset(
    {
        "gate-failed",
        "Indeterminate",  # classify_route canonical
        "indeterminate",  # legacy alias
        "INCONCLUSIVE",
        "H1-indeterminate",
    }
)


@dataclass(frozen=True)
class Interval:
    lower: float
    upper: float
    point: float

    def entirely_above(self, thr: float) -> bool:
        return self.lower >= thr

    def entirely_below(self, thr: float) -> bool:
        return self.upper < thr


def mean_contrast(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(a) - np.mean(b))


def bootstrap_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    index_draws: np.ndarray,
    *,
    q: list[float],
) -> Interval:
    stats = []
    for row in index_draws:
        stats.append(float(np.mean(a[row]) - np.mean(b[row])))
    arr = np.asarray(stats, dtype=np.float64)
    lo, hi = percentile_linear(arr, q)
    return Interval(float(lo), float(hi), float(np.mean(a) - np.mean(b)))


def bootstrap_ratio_of_means(
    num: np.ndarray,
    den: np.ndarray,
    index_draws: np.ndarray,
    *,
    q: list[float] = [5.0, 95.0],
) -> Interval:
    stats = []
    for row in index_draws:
        stats.append(ratio_of_paired_means(num[row], den[row]))
    arr = np.asarray(stats, dtype=np.float64)
    lo, hi = percentile_linear(arr, q)
    return Interval(float(lo), float(hi), ratio_of_paired_means(num, den))


@dataclass
class RouteContrasts:
    m_TT: np.ndarray
    m_CT: np.ndarray
    m_TC: np.ndarray
    m_CC: np.ndarray

    @property
    def T(self) -> float:
        return mean_contrast(self.m_TT, self.m_CC)

    @property
    def E_post_fwd(self) -> float:
        return mean_contrast(self.m_TT, self.m_CT)

    @property
    def E_post_rev(self) -> float:
        return mean_contrast(self.m_TC, self.m_CC)

    def n_post(self) -> float:
        t = self.T
        if t == 0.0:
            raise RuntimeError("T==0; refuse route ratios")
        return self.E_post_fwd / t

    def s_post(self) -> float:
        t = self.T
        if t == 0.0:
            raise RuntimeError("T==0; refuse route ratios")
        return self.E_post_rev / t


def classify_route(
    n_post_iv: Interval,
    s_post_iv: Interval,
    *,
    theta_hi: float = THETA_HI,
    theta_lo: float = THETA_LO,
) -> RouteClass:
    moved = n_post_iv.entirely_above(theta_hi) and s_post_iv.entirely_above(theta_hi)
    rand = n_post_iv.entirely_above(theta_hi) and s_post_iv.entirely_below(theta_lo)
    ror = n_post_iv.entirely_below(theta_lo) and s_post_iv.entirely_above(theta_hi)
    rtag = n_post_iv.entirely_below(theta_lo) and s_post_iv.entirely_below(theta_lo)
    if moved:
        return "R-moved"
    if rand:
        return "R-and"
    if ror:
        return "R-or"
    if rtag:
        return "R-tag"
    # R-intermediate: for every named class, at least one defining interval is
    # entirely on the wrong side
    named_wrong = []
    # wrong for R-moved: not (n>=hi and s>=hi) with entire-side failures
    if n_post_iv.entirely_below(theta_hi) or s_post_iv.entirely_below(theta_hi):
        named_wrong.append("R-moved")
    if n_post_iv.entirely_below(theta_hi) or s_post_iv.entirely_above(theta_lo):
        named_wrong.append("R-and")
    if n_post_iv.entirely_above(theta_lo) or s_post_iv.entirely_below(theta_hi):
        named_wrong.append("R-or")
    if n_post_iv.entirely_above(theta_lo) or s_post_iv.entirely_above(theta_lo):
        named_wrong.append("R-tag")
    if len(set(named_wrong)) == 4:
        return "R-intermediate"
    return "Indeterminate"


@dataclass
class ModelGates:
    trigger_asr: float
    clean_false_fire: float
    first_token_validity: float
    eligible_fraction: float
    T_point: float
    T_lower_95: float
    passed: bool
    failures: list[str] = field(default_factory=list)


def evaluate_model_gates(
    *,
    trigger_asr: float,
    clean_false_fire: float,
    first_token_validity: float,
    n_eligible: int,
    n_frozen: int,
    T_interval_95: Interval,
) -> ModelGates:
    failures: list[str] = []
    elig = n_eligible / n_frozen if n_frozen else 0.0
    if trigger_asr < ASR_MIN:
        failures.append(f"trigger_asr {trigger_asr}<{ASR_MIN}")
    if clean_false_fire > FALSE_FIRE_MAX:
        failures.append(f"clean_false_fire {clean_false_fire}>{FALSE_FIRE_MAX}")
    if first_token_validity < FIRST_TOKEN_MIN:
        failures.append(f"first_token_validity {first_token_validity}<{FIRST_TOKEN_MIN}")
    if elig < ELIGIBILITY_MIN_FRAC:
        failures.append(f"eligible_fraction {elig}<{ELIGIBILITY_MIN_FRAC}")
    if T_interval_95.lower <= T_GATE_NAT:
        failures.append(f"T_lower_95 {T_interval_95.lower}<={T_GATE_NAT}")
    return ModelGates(
        trigger_asr=trigger_asr,
        clean_false_fire=clean_false_fire,
        first_token_validity=first_token_validity,
        eligible_fraction=elig,
        T_point=T_interval_95.point,
        T_lower_95=T_interval_95.lower,
        passed=len(failures) == 0,
        failures=failures,
    )


def pair_effect_forward(m_TT: np.ndarray, m_patched: np.ndarray) -> np.ndarray:
    return m_TT - m_patched


def pair_effect_reverse(m_patched: np.ndarray, m_CC: np.ndarray) -> np.ndarray:
    return m_patched - m_CC


def rank_pairs_signed(
    effects_by_pair: dict[tuple[int, int], float],
) -> list[tuple[tuple[int, int], float, int]]:
    """Descending signed effect; tie-break (head15, head16) ascending."""
    items = sorted(
        effects_by_pair.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )
    out = []
    for rank_i, (pair, eff) in enumerate(items, start=1):
        out.append((pair, eff, rank_i))
    return out


def bootstrap_core_rank(
    per_prompt_effects: dict[tuple[int, int], np.ndarray],
    core: tuple[int, int],
    index_draws: np.ndarray,
) -> Interval:
    ranks = []
    for row in index_draws:
        means = {p: float(np.mean(v[row])) for p, v in per_prompt_effects.items()}
        ranking = rank_pairs_signed(means)
        rank_map = {p: r for p, _, r in ranking}
        ranks.append(rank_map[core])
    arr = np.asarray(ranks, dtype=np.int64)
    lo, hi = percentile_rank_inverted_cdf(arr.astype(float), [5.0, 95.0])
    point = rank_pairs_signed({p: float(np.mean(v)) for p, v in per_prompt_effects.items()})
    point_rank = next(r for p, _, r in point if p == core)
    return Interval(float(lo), float(hi), float(point_rank))


def evaluate_h1_direction(
    *,
    route_class: RouteClass,
    direction: Literal["forward", "reverse"],
    rank_iv: Interval,
    share_iv: Interval,
    marginal_h15_iv: Interval,
    marginal_h16_iv: Interval,
    route_denom_ok: bool,
) -> dict[str, Any]:
    """H1 directional decision with frozen three-way states only.

    Decision states: ``H1-pass`` / ``H1-fail`` / ``H1-indeterminate`` (or
    ``inapplicable``). When rank+share pass but marginal contribution does not
    establish a pass, set annotation ``unresolved_marginal_contribution=True``;
    that annotation never replaces the decision state. An entire marginal
    interval below 0.1 is a decisive ``H1-fail``.
    """
    applicable = False
    if route_class == "R-moved":
        applicable = True
    elif route_class == "R-and" and direction == "forward":
        applicable = True
    elif route_class == "R-or" and direction == "reverse":
        applicable = True
    if not applicable or not route_denom_ok:
        return {"state": "inapplicable", "direction": direction, "route_class": route_class}

    pass_rank = rank_iv.upper <= H1_RANK_UPPER
    pass_share_strong = share_iv.lower >= H1_SHARE_STRONG
    pass_share_partial = share_iv.lower >= H1_SHARE_PARTIAL
    pass_share = pass_share_strong or pass_share_partial
    pass_marg = (
        marginal_h15_iv.lower >= H1_MARGINAL and marginal_h16_iv.lower >= H1_MARGINAL
    )
    rank_fail = rank_iv.lower > H1_RANK_UPPER
    share_fail = share_iv.upper < H1_SHARE_PARTIAL
    # Decisive marginal failure: entire interval below threshold on either head
    marg_fail = (
        marginal_h15_iv.upper < H1_MARGINAL or marginal_h16_iv.upper < H1_MARGINAL
    )
    share_label = "strong" if pass_share_strong else ("partial" if pass_share_partial else None)
    unresolved = bool(pass_rank and pass_share and not pass_marg)

    def _annotate(out: dict[str, Any]) -> dict[str, Any]:
        if unresolved:
            out["unresolved_marginal_contribution"] = True
            out["share"] = share_label
            out["marginal_h15_upper"] = marginal_h15_iv.upper
            out["marginal_h16_upper"] = marginal_h16_iv.upper
        return out

    # Decisive failures (rank / share / entire-side marginal)
    if rank_fail or share_fail or marg_fail:
        return _annotate({"state": "H1-fail", "direction": direction})

    if pass_rank and pass_share and pass_marg:
        return {
            "state": "H1-pass",
            "direction": direction,
            "share": share_label,
            "unresolved_marginal_contribution": False,
        }

    # Not pass, no decisive failure → indeterminate (may carry unresolved annotation)
    return _annotate({"state": "H1-indeterminate", "direction": direction})


def combine_h1_rmoved(fwd: dict[str, Any], rev: dict[str, Any]) -> dict[str, Any]:
    """R-moved full H1: both directions pass; fail if either H1-fail.

    S2: one direction H1-pass and the other H1-indeterminate → full_H1 =
    H1-indeterminate, with the passing direction recorded as directional/partial
    evidence (not elevated to full H1-directional).
    """
    fs, rs = fwd.get("state"), rev.get("state")
    if fs == "H1-fail" or rs == "H1-fail":
        return {"full_H1": "H1-fail", "forward": fwd, "reverse": rev}
    if fs == "H1-pass" and rs == "H1-pass":
        return {"full_H1": "H1-pass", "forward": fwd, "reverse": rev}
    # Pass + indeterminate → full H1-indeterminate; record directional evidence
    if {fs, rs} == {"H1-pass", "H1-indeterminate"}:
        passing = "forward" if fs == "H1-pass" else "reverse"
        return {
            "full_H1": "H1-indeterminate",
            "directional_partial_evidence": {
                "direction": passing,
                "state": "H1-pass",
                "detail": "Single-direction pass under R-moved; other direction not pass",
            },
            "forward": fwd,
            "reverse": rev,
        }
    if fs == "H1-pass" or rs == "H1-pass":
        passing = "forward" if fs == "H1-pass" else "reverse"
        return {
            "full_H1": "H1-indeterminate",
            "directional_partial_evidence": {"direction": passing, "state": "H1-pass"},
            "forward": fwd,
            "reverse": rev,
        }
    return {"full_H1": "H1-indeterminate", "forward": fwd, "reverse": rev}


def crossfit_topk_selection(
    head_effects_fold: dict[tuple[int, int], float],
    k: int,
) -> list[tuple[int, int]]:
    """head keys as (layer, head_index); rank by signed effect desc."""
    ranked = sorted(
        head_effects_fold.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1])
    )
    return [p for p, _ in ranked[:k]]


def evaluate_h2_h3(
    *,
    k_star_le_2: bool,
    k_star_gt_10: bool,
    h1_state: str,
    all_component_share_upper_below_025: bool,
    top2_set_conservation_seeds: int,
    top2_varies_across_seeds: bool,
) -> dict[str, Any]:
    out: dict[str, Any] = {"H2": None, "H3a": None, "H3b": None}
    if k_star_gt_10 and all_component_share_upper_below_025:
        out["H2"] = "supported"
    elif k_star_gt_10:
        out["H2"] = "indeterminate"
    else:
        out["H2"] = "not_applicable_or_not_met"
    if k_star_le_2 and h1_state == "H1-fail":
        if top2_set_conservation_seeds >= 4:
            out["H3a"] = "supported"
        if top2_varies_across_seeds:
            out["H3b"] = "supported"
    return out


def exploratory_stop_triggered(single_bos_ranks: Sequence[int]) -> bool:
    """Stop if pair ranks worse than 10th in ≥2 of 3 exploratory seeds."""
    worse = sum(1 for r in single_bos_ranks if r > EXPLORATORY_STOP_RANK)
    return worse >= 2


def normalize_study_label(label: str) -> str:
    """Map aliases to canonical study-level vocabulary."""
    if label in {"indeterminate", "Indeterminate"}:
        return "Indeterminate"
    return label


def study_level_decision(seed_labels: Sequence[str]) -> str:
    """≥2/5 gate-failed or Indeterminate → INCONCLUSIVE; else ≥4/5 same → SUPPORTED.

    S1: ``classify_route``'s exact ``\"Indeterminate\"`` counts toward inconclusive.
    """
    n = len(seed_labels)
    if n != 5:
        raise ValueError("Primary study-level decision expects exactly 5 seeds")
    labels = [normalize_study_label(s) for s in seed_labels]
    incon = sum(1 for s in labels if s in STUDY_INCONCLUSIVE_LABELS)
    if incon >= 2:
        return "INCONCLUSIVE"
    from collections import Counter

    c = Counter(labels)
    label, count = c.most_common(1)[0]
    if count >= 4:
        return f"SUPPORTED_WITHIN_FAMILY:{label}"
    return "MIXED"


def secondary_l1523_replication_criteria(
    *,
    rank_upper_90: float,
    e_pair_over_T_lower_90: float,
    marg15_lower_90: float,
    marg16_lower_90: float,
) -> dict[str, Any]:
    return {
        "rank_ok": rank_upper_90 <= 3,
        "share_ok": e_pair_over_T_lower_90 >= 0.25,
        "marginal15_ok": marg15_lower_90 >= 0.1,
        "marginal16_ok": marg16_lower_90 >= 0.1,
        "pass": (
            rank_upper_90 <= 3
            and e_pair_over_T_lower_90 >= 0.25
            and marg15_lower_90 >= 0.1
            and marg16_lower_90 >= 0.1
        ),
    }


def run_route_analysis(
    contrasts: RouteContrasts,
    draws: F1Draws,
) -> dict[str, Any]:
    t_iv = bootstrap_mean_diff(
        contrasts.m_TT, contrasts.m_CC, draws.full_set, q=[2.5, 97.5]
    )
    # n_post = E_post_fwd / T as ratio of means under bootstrap of prompt means
    e_fwd = contrasts.m_TT - contrasts.m_CT
    e_rev = contrasts.m_TC - contrasts.m_CC
    t_vec = contrasts.m_TT - contrasts.m_CC
    n_iv = bootstrap_ratio_of_means(e_fwd, t_vec, draws.full_set)
    s_iv = bootstrap_ratio_of_means(e_rev, t_vec, draws.full_set)
    route = classify_route(n_iv, s_iv)
    return {
        "T_interval_95": t_iv,
        "n_post_90": n_iv,
        "s_post_90": s_iv,
        "route_class": route,
        "E_post_fwd": contrasts.E_post_fwd,
        "E_post_rev": contrasts.E_post_rev,
        "interaction": contrasts.n_post() - contrasts.s_post() if contrasts.T != 0 else None,
    }


def prepare_f1_for_model(
    eligible_ids_in_allocation_order: Sequence[Any],
    held_out_ids_in_allocation_order: Sequence[Any],
) -> F1Draws:
    folds = assign_f2_folds(held_out_ids_in_allocation_order)
    elig = list(eligible_ids_in_allocation_order)
    elig_set = set(elig)
    n_a = sum(1 for i in folds.fold_a_ids if i in elig_set)
    n_b = sum(1 for i in folds.fold_b_ids if i in elig_set)
    return draw_f1_resamples(len(elig), n_a, n_b)
