"""Owner-approved H3 applicable-direction (H3-B) — synthetic, no forwards."""

from __future__ import annotations

from traces.q1.analysis import evaluate_h2_h3, study_level_decision
from traces.q1.analysis_ext import (
    assess_unit_h3_direction_support,
    h3_study_level_from_seed_records,
    h3_top2_conservation,
    top2_heads_from_mean_effects,
)
from traces.q1.component_sets import EXPECTED_DOWNSTREAM_HEADS, EXPECTED_UPSTREAM_HEADS
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256
import numpy as np

F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"
PAIR_A = frozenset([(15, 6), (16, 5)])
PAIR_B = frozenset([(14, 0), (14, 1)])
PAIR_C = frozenset([(13, 0), (13, 1)])
ROSTER = [f"primary_l19_seed{s}" for s in (42, 43, 44, 45, 46)]


def _k(ok: bool) -> dict:
    return {
        "k_star_le_2": ok,
        "k_star_gt_10": False,
        "definitive_refused": False,
        "undetermined_concentration": not ok,
    }


def _effects_with_top2(top2: frozenset[tuple[int, int]], n: int = 8) -> dict:
    """Complete upstream map whose signed means realize ``top2``."""
    fx = {h: np.full(n, 0.01, dtype=np.float64) for h in EXPECTED_UPSTREAM_HEADS}
    ranked = sorted(top2)
    fx[ranked[0]] = np.full(n, 2.0)
    fx[ranked[1]] = np.full(n, 1.5)
    return fx


def _eligible_rec(top2: frozenset[tuple[int, int]], *, route: str = "R-and") -> dict:
    return {
        "full_h3_eligible": True,
        "top2_for_h3": top2,
        "route_class": route,
        "support_class": "full",
        "reason": None,
    }


def _non_rec(reason: str, **extra) -> dict:
    return {
        "full_h3_eligible": False,
        "top2_for_h3": None,
        "support_class": "non_supporting",
        "reason": reason,
        **extra,
    }


# --- Per-seed direction -------------------------------------------------------


def test_r_and_uses_forward_top2_only():
    fwd = _effects_with_top2(PAIR_A)
    rev = _effects_with_top2(PAIR_B)
    out = assess_unit_h3_direction_support(
        route_class="R-and",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=fwd,
        head_effects_rev=rev,
        crossfit_fwd={"k_star": _k(True)},
        crossfit_rev={"k_star": _k(True)},
        n_eligible=8,
    )
    assert out["full_h3_eligible"] is True
    assert out["top2_for_h3"] == PAIR_A
    assert out["applicable_directions"] == ["forward"]


def test_r_or_uses_reverse_even_if_forward_differs():
    fwd = _effects_with_top2(PAIR_A)
    rev = _effects_with_top2(PAIR_B)
    out = assess_unit_h3_direction_support(
        route_class="R-or",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=fwd,
        head_effects_rev=rev,
        crossfit_fwd={"k_star": _k(True)},
        crossfit_rev={"k_star": _k(True)},
        n_eligible=8,
    )
    assert out["full_h3_eligible"] is True
    assert out["top2_for_h3"] == PAIR_B
    assert out["applicable_directions"] == ["reverse"]


def test_mixed_cohort_no_favorable_direction_selection():
    """Each seed keeps its own applicable direction; no cross-seed favor picking."""
    records = {
        ROSTER[0]: _eligible_rec(PAIR_A, route="R-and"),
        ROSTER[1]: _eligible_rec(PAIR_A, route="R-or"),
        ROSTER[2]: _eligible_rec(PAIR_A, route="R-and"),
        ROSTER[3]: _eligible_rec(PAIR_A, route="R-or"),
        ROSTER[4]: _eligible_rec(PAIR_B, route="R-and"),
    }
    # Verify assess assigns directions correctly for mixed routes
    a = assess_unit_h3_direction_support(
        route_class="R-and",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=False,
        head_effects_fwd=_effects_with_top2(PAIR_A),
        head_effects_rev=_effects_with_top2(PAIR_B),
        crossfit_fwd={"k_star": _k(True)},
        crossfit_rev={"k_star": _k(False)},
        n_eligible=8,
    )
    o = assess_unit_h3_direction_support(
        route_class="R-or",
        h1_full_state="H1-fail",
        denom_fwd_ok=False,
        denom_rev_ok=True,
        head_effects_fwd=_effects_with_top2(PAIR_A),
        head_effects_rev=_effects_with_top2(PAIR_B),
        crossfit_fwd={"k_star": _k(False)},
        crossfit_rev={"k_star": _k(True)},
        n_eligible=8,
    )
    assert a["top2_for_h3"] == PAIR_A
    assert o["top2_for_h3"] == PAIR_B
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["n_qualifying"] == 5
    assert study["H3a"] == "supported"
    assert study["H3a_count"] == 4


def test_rmoved_agreeing_top2_eligible():
    fx = _effects_with_top2(PAIR_A)
    out = assess_unit_h3_direction_support(
        route_class="R-moved",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=fx,
        head_effects_rev=fx,
        crossfit_fwd={"k_star": _k(True)},
        crossfit_rev={"k_star": _k(True)},
        n_eligible=8,
    )
    assert out["full_h3_eligible"] is True
    assert out["rmoved_top2_agree"] is True
    assert out["top2_for_h3"] == PAIR_A


def test_rmoved_disagreeing_top2_non_supporting_reports_both():
    out = assess_unit_h3_direction_support(
        route_class="R-moved",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=_effects_with_top2(PAIR_A),
        head_effects_rev=_effects_with_top2(PAIR_B),
        crossfit_fwd={"k_star": _k(True)},
        crossfit_rev={"k_star": _k(True)},
        n_eligible=8,
    )
    assert out["full_h3_eligible"] is False
    assert out["support_class"] == "directional_partial"
    assert out["top2_forward"] == sorted(PAIR_A)
    assert out["top2_reverse"] == sorted(PAIR_B)


def test_rmoved_one_direction_k_star_fails():
    fx = _effects_with_top2(PAIR_A)
    out = assess_unit_h3_direction_support(
        route_class="R-moved",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=fx,
        head_effects_rev=fx,
        crossfit_fwd={"k_star": _k(True)},
        crossfit_rev={"k_star": _k(False)},
        n_eligible=8,
    )
    assert out["full_h3_eligible"] is False
    assert "reverse k*" in (out["reason"] or "")


def test_rmoved_incomplete_ranking_or_denom_fails():
    fx = _effects_with_top2(PAIR_A)
    incomplete = {h: fx[h] for h in list(EXPECTED_UPSTREAM_HEADS)[:100]}
    out = assess_unit_h3_direction_support(
        route_class="R-moved",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=False,
        head_effects_fwd=incomplete,
        head_effects_rev=fx,
        crossfit_fwd={"k_star": _k(True)},
        crossfit_rev={"k_star": _k(True)},
        n_eligible=8,
    )
    assert out["full_h3_eligible"] is False
    assert out["top2_forward"] is None or "denominator" in (out["reason"] or "") or "ranking" in (
        out["reason"] or ""
    )


def test_h1_indeterminate_and_pass_never_support():
    fx = _effects_with_top2(PAIR_A)
    for state in ("H1-indeterminate", "H1-pass"):
        out = assess_unit_h3_direction_support(
            route_class="R-and",
            h1_full_state=state,
            denom_fwd_ok=True,
            denom_rev_ok=True,
            head_effects_fwd=fx,
            head_effects_rev=fx,
            crossfit_fwd={"k_star": _k(True)},
            crossfit_rev={"k_star": _k(True)},
            n_eligible=8,
        )
        assert out["full_h3_eligible"] is False
        assert out["support_class"] == "h1_not_fail"


def test_route_inapplicable_no_synthetic_default_pair():
    out = assess_unit_h3_direction_support(
        route_class="R-tag",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=_effects_with_top2(PAIR_A),
        head_effects_rev=_effects_with_top2(PAIR_A),
        crossfit_fwd={"k_star": _k(True)},
        crossfit_rev={"k_star": _k(True)},
        n_eligible=8,
    )
    assert out["full_h3_eligible"] is False
    assert out["top2_for_h3"] is None
    assert out["support_class"] == "route_inapplicable"


# --- Study aggregation --------------------------------------------------------


def test_four_qualifying_one_inapplicable_preserves_h3a():
    records = {sid: _eligible_rec(PAIR_A) for sid in ROSTER[:4]}
    records[ROSTER[4]] = _non_rec("R-or denom failed", route_class="R-or")
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["n_qualifying"] == 4
    assert study["H3a"] == "supported"
    assert study["H3a_count"] == 4
    assert study["varies"] is False
    assert study["H3b"] is None
    assert ROSTER[4] in study["non_supporting"]


def test_inapplicable_fifth_cannot_create_spurious_h3b():
    records = {sid: _eligible_rec(PAIR_A) for sid in ROSTER[:4]}
    records[ROSTER[4]] = _non_rec(
        "inapplicable",
        top2_forward=sorted(PAIR_B),
        top2_reverse=sorted(PAIR_C),
    )
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["varies"] is False
    assert study["H3b"] is None


def test_three_qualifying_no_full_h3_claim():
    records = {sid: _eligible_rec(PAIR_A) for sid in ROSTER[:3]}
    records[ROSTER[3]] = _non_rec("gate-failed")
    records[ROSTER[4]] = _non_rec("Indeterminate")
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["n_qualifying"] == 3
    assert study["H3a"] is None
    assert study["H3b"] is None
    assert study["H3a_eligible"] is False


def test_h3a_4plus1_excludes_h3b_dual_support():
    """H3-A: 4/5 conserved pair → H3a only; minority disclosed, H3b not supported."""
    records = {
        ROSTER[0]: _eligible_rec(PAIR_A),
        ROSTER[1]: _eligible_rec(PAIR_A),
        ROSTER[2]: _eligible_rec(PAIR_A),
        ROSTER[3]: _eligible_rec(PAIR_A),
        ROSTER[4]: _eligible_rec(PAIR_B),
    }
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["H3a"] == "supported"
    assert study["H3b"] is None
    assert study["H3b_eligible"] is False
    assert study["H3a_count"] == 4
    assert study["varies"] is True
    assert study["minority_pairs"] is not None
    assert ROSTER[4] in study["minority_pairs"]


def test_study_inconclusive_precedence_blocks_h3():
    records = {sid: _eligible_rec(PAIR_A) for sid in ROSTER}
    labels = ["gate-failed", "Indeterminate", "R-and", "R-and", "R-and"]
    assert study_level_decision(labels) == "INCONCLUSIVE"
    study = h3_study_level_from_seed_records(
        records, expected_roster=ROSTER, study_inconclusive=True
    )
    assert study["study_inconclusive_precedence"] is True
    assert study["H3a"] is None
    assert study["H3b"] is None
    assert study["H3a_eligible"] is False


def test_legacy_h3_top2_conservation_five_sets_and_partial():
    top2 = {s: PAIR_A for s in ("s1", "s2", "s3", "s4")}
    top2["s5"] = PAIR_B
    cons = h3_top2_conservation(top2, require_five_seeds=True)
    assert cons["H3a_eligible"] is True
    assert cons["H3a_count"] == 4
    assert cons["varies"] is True
    assert cons.get("H3a") == "supported"
    assert cons.get("H3b") is None  # H3-A mutual exclusion
    partial = {s: PAIR_A for s in ("s1", "s2", "s3")}
    bad = h3_top2_conservation(partial, require_five_seeds=True)
    assert bad["H3a_eligible"] is False


def test_signed_ranking_tiebreak_ignores_downstream():
    n = 4
    fx = {h: np.full(n, 0.0) for h in EXPECTED_UPSTREAM_HEADS}
    fx[(10, 0)] = np.full(n, 3.0)
    fx[(10, 1)] = np.full(n, 2.0)
    # Plant stronger downstream heads — must not enter top-2
    for h in list(EXPECTED_DOWNSTREAM_HEADS)[:2]:
        fx[h] = np.full(n, 9.0)
    # Filter path used by assess: only upstream
    upstream = {k: v for k, v in fx.items() if k in EXPECTED_UPSTREAM_HEADS}
    top2 = top2_heads_from_mean_effects(
        upstream, require_complete_upstream=True, n_eligible=n
    )
    assert top2 == frozenset([(10, 0), (10, 1)])
    assert top2.isdisjoint(EXPECTED_DOWNSTREAM_HEADS)


def test_evaluate_h2_h3_h1_indeterminate_blocks():
    no_h3 = evaluate_h2_h3(
        k_star_le_2=True,
        k_star_gt_10=False,
        h1_state="H1-indeterminate",
        all_component_share_upper_below_025=False,
        top2_set_conservation_seeds=5,
        top2_varies_across_seeds=True,
    )
    assert no_h3["H3a"] is None
    assert no_h3["H3b"] is None


def test_f1_preserved():
    assert f1_full_set_stream_sha256() == F1_REF
