"""H3-A / H3-B / H3-C owner-approved corrections — synthetic, no forwards."""

from __future__ import annotations

from traces.q1.analysis import study_level_decision
from traces.q1.analysis_ext import (
    assess_unit_h3_direction_support,
    h3_study_level_from_seed_records,
)
from traces.q1.component_sets import EXPECTED_UPSTREAM_HEADS
from traces.q1.analysis_ext import selection_key
from traces.q1.driver import ModelUnitInput, PRIMARY_ROLE, run_study
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256
import numpy as np

F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"
PAIR_A = frozenset([(15, 6), (16, 5)])
PAIR_B = frozenset([(14, 0), (14, 1)])
PAIR_C = frozenset([(13, 0), (13, 1)])
ROSTER = [f"primary_l19_seed{s}" for s in (42, 43, 44, 45, 46)]


def _eligible(top2=PAIR_A, **kw):
    return {
        "full_h3_eligible": True,
        "top2_for_h3": top2,
        "h3_endpoint_status": "full_qualifying",
        "support_class": "full",
        "route_class": kw.get("route_class", "R-and"),
        "reason": None,
    }


def _indeterminate(reason="H1-indeterminate", **kw):
    return {
        "full_h3_eligible": False,
        "top2_for_h3": None,
        "h3_endpoint_status": "h3_indeterminate",
        "support_class": kw.get("support_class", "h1_not_fail"),
        "reason": reason,
        "uncertainty_kind": kw.get("uncertainty_kind", "h1_indeterminate"),
        "h1_state": kw.get("h1_state", "H1-indeterminate"),
        "route_class": kw.get("route_class", "R-and"),
    }


def _gate_failed(reason="gate-failed"):
    return {
        "full_h3_eligible": False,
        "top2_for_h3": None,
        "h3_endpoint_status": "gate_failed",
        "support_class": "gate_failed",
        "gates_passed": False,
        "reason": reason,
    }


def _non_support(reason="R-tag", **kw):
    return {
        "full_h3_eligible": False,
        "top2_for_h3": None,
        "h3_endpoint_status": "determinate_non_support",
        "support_class": kw.get("support_class", "route_inapplicable"),
        "reason": reason,
        "route_class": kw.get("route_class", "R-tag"),
    }


def _effects(top2, n=8):
    fx = {h: np.full(n, 0.01) for h in EXPECTED_UPSTREAM_HEADS}
    ranked = sorted(top2)
    fx[ranked[0]] = np.full(n, 2.0)
    fx[ranked[1]] = np.full(n, 1.5)
    return fx


def _k(band: str) -> dict:
    if band == "le_2":
        return {"k_star_le_2": True, "k_star_gt_10": False, "definitive_refused": False}
    if band == "gt_10":
        return {"k_star_le_2": False, "k_star_gt_10": True, "definitive_refused": False}
    if band == "undetermined":
        return {
            "k_star_le_2": False,
            "k_star_gt_10": False,
            "undetermined_concentration": True,
            "definitive_refused": False,
        }
    return {
        "k_star_le_2": False,
        "k_star_gt_10": False,
        "definitive_refused": True,
        "reason": "incomplete",
    }


# --- H3-A --------------------------------------------------------------------


def test_h3a_5of5_identical_excludes_h3b():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER}
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["H3a"] == "supported"
    assert study["H3b"] is None
    assert study["varies"] is False


def test_h3a_4plus1_excludes_h3b():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER[:4]}
    records[ROSTER[4]] = _eligible(PAIR_B)
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["H3a"] == "supported"
    assert study["H3b"] is None
    assert study["H3b_eligible"] is False
    assert study["minority_pairs"][ROSTER[4]] == sorted(PAIR_B)


def test_h3b_3plus2_among_five_qualifying():
    records = {
        ROSTER[0]: _eligible(PAIR_A),
        ROSTER[1]: _eligible(PAIR_A),
        ROSTER[2]: _eligible(PAIR_A),
        ROSTER[3]: _eligible(PAIR_B),
        ROSTER[4]: _eligible(PAIR_C),
    }
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["n_qualifying"] == 5
    assert study["H3a_count"] == 3
    assert study["H3a"] is None
    assert study["H3b"] == "supported"


def test_h3a_four_same_one_inapplicable():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER[:4]}
    records[ROSTER[4]] = _non_support("R-intermediate", route_class="R-intermediate")
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["H3a"] == "supported"
    assert study["H3b"] is None
    assert study["endpoint_counts"]["determinate_non_support"] == 1


# --- H3-B endpoint uncertainty -----------------------------------------------


def test_one_h3_indeterminate_four_h3a_still_supports():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER[:4]}
    records[ROSTER[4]] = _indeterminate()
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["endpoint_counts"]["h3_indeterminate"] == 1
    assert study["h3_endpoint_inconclusive"] is False
    assert study["H3a"] == "supported"
    assert study["H3b"] is None


def test_two_h1_indeterminate_h3_inconclusive():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER[:3]}
    records[ROSTER[3]] = _indeterminate("H1-indeterminate a")
    records[ROSTER[4]] = _indeterminate("H1-indeterminate b")
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["h3_endpoint_inconclusive"] is True
    assert study["H3a"] is None and study["H3b"] is None
    assert study["h3_endpoint_decision"] == "INCONCLUSIVE"


def test_two_kstar_undetermined_h3_inconclusive():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER[:3]}
    records[ROSTER[3]] = _indeterminate(
        "k* undetermined",
        support_class="k_star_undetermined",
        uncertainty_kind="k_star_undetermined",
    )
    records[ROSTER[4]] = _indeterminate(
        "k* undetermined 2",
        support_class="k_star_undetermined",
        uncertainty_kind="k_star_undetermined",
    )
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["h3_endpoint_inconclusive"] is True
    assert study["H3a"] is None


def test_two_gate_failures_h3_inconclusive():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER[:3]}
    records[ROSTER[3]] = _gate_failed()
    records[ROSTER[4]] = _gate_failed("gate-failed 2")
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["endpoint_counts"]["gate_failed"] == 2
    assert study["h3_endpoint_inconclusive"] is True
    assert study["H3a"] is None


def test_two_determinate_inapplicable_not_false_inconclusive():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER[:3]}
    records[ROSTER[3]] = _non_support("R-tag", route_class="R-tag")
    records[ROSTER[4]] = _non_support("R-intermediate", route_class="R-intermediate")
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["h3_endpoint_inconclusive"] is False
    assert study["H3a"] is None  # only 3 qualifying
    assert study["h3_endpoint_decision"] == "NOT_SUPPORTED"
    assert study["endpoint_counts"]["determinate_non_support"] == 2


def test_route_inconclusive_blocks_h3_even_with_four_conserved():
    records = {sid: _eligible(PAIR_A) for sid in ROSTER}
    assert study_level_decision(
        ["gate-failed", "Indeterminate", "R-and", "R-and", "R-and"]
    ) == "INCONCLUSIVE"
    study = h3_study_level_from_seed_records(
        records, expected_roster=ROSTER, study_inconclusive=True
    )
    assert study["h3_endpoint_inconclusive"] is True
    assert study["H3a"] is None and study["H3b"] is None


def test_rmoved_disagreement_is_determinate_non_support():
    out = assess_unit_h3_direction_support(
        route_class="R-moved",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=_effects(PAIR_A),
        head_effects_rev=_effects(PAIR_B),
        crossfit_fwd={"k_star": _k("le_2")},
        crossfit_rev={"k_star": _k("le_2")},
        n_eligible=8,
    )
    assert out["h3_endpoint_status"] == "determinate_non_support"
    assert out["full_h3_eligible"] is False
    assert out["top2_forward"] == sorted(PAIR_A)
    assert out["top2_reverse"] == sorted(PAIR_B)


def test_kstar_undetermined_is_h3_indeterminate():
    out = assess_unit_h3_direction_support(
        route_class="R-and",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=_effects(PAIR_A),
        head_effects_rev=_effects(PAIR_A),
        crossfit_fwd={"k_star": _k("undetermined")},
        crossfit_rev={"k_star": _k("le_2")},
        n_eligible=8,
    )
    assert out["h3_endpoint_status"] == "h3_indeterminate"
    assert out["uncertainty_kind"] == "k_star_undetermined"


def test_route_indeterminate_is_h3_indeterminate_not_determinate():
    out = assess_unit_h3_direction_support(
        route_class="Indeterminate",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=_effects(PAIR_A),
        head_effects_rev=_effects(PAIR_A),
        crossfit_fwd={"k_star": _k("le_2")},
        crossfit_rev={"k_star": _k("le_2")},
        n_eligible=8,
    )
    assert out["h3_endpoint_status"] == "h3_indeterminate"


def test_kstar_gt10_determinate_non_support():
    out = assess_unit_h3_direction_support(
        route_class="R-and",
        h1_full_state="H1-fail",
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=_effects(PAIR_A),
        head_effects_rev=_effects(PAIR_A),
        crossfit_fwd={"k_star": _k("gt_10")},
        crossfit_rev={"k_star": _k("le_2")},
        n_eligible=8,
    )
    assert out["h3_endpoint_status"] == "determinate_non_support"


# --- H3-C fixed roster via run_study ----------------------------------------


def _joint_for_topk(head_fx: dict, e_route: np.ndarray, k: int) -> dict:
    means = {h: float(np.mean(v)) for h, v in head_fx.items()}
    ranked = sorted(means.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    out = {}
    for kk in range(1, k + 1):
        sel = [h for h, _ in ranked[:kk]]
        key = selection_key(sel)
        acc = np.zeros_like(e_route, dtype=np.float64)
        for h in sel:
            acc = acc + head_fx[h]
        out[key] = acc
    return out


def _tiny_primary(seed: int, *, name: str | None = None, n: int = 40) -> ModelUnitInput:
    rng = np.random.default_rng(seed)
    ids = [f"p{seed}-{i}" for i in range(n)]
    m_CC = rng.normal(0, 0.01, n)
    m_TT = m_CC + 3.0
    m_CT = m_CC + 0.1
    m_TC = m_TT - 0.1
    pair_fwd = {}
    pair_rev = {}
    for h15 in range(8):
        for h16 in range(8):
            base = rng.normal(0.2, 0.05, n)
            if (h15, h16) == (6, 5):
                base = base + 1.5
            pair_fwd[(h15, h16)] = base
            pair_rev[(h15, h16)] = base * 0.9
    head_fwd = {(L, h): rng.normal(0.05, 0.01, n) for L in range(0, 19) for h in range(8)}
    head_fwd[(15, 6)] = np.full(n, 0.9)
    head_fwd[(16, 5)] = np.full(n, 0.85)
    head_rev = {k: v * 0.9 for k, v in head_fwd.items()}
    e_route = m_TT - m_CT
    return ModelUnitInput(
        model_unit=name or f"primary_l19_seed{seed}",
        family="l19",
        role=PRIMARY_ROLE,
        prompt_ids=ids,
        m_TT=m_TT,
        m_CT=m_CT,
        m_TC=m_TC,
        m_CC=m_CC,
        eligible_mask=np.ones(n, dtype=bool),
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.95,
        n_frozen=n,
        pair_effects_fwd=pair_fwd,
        pair_effects_rev=pair_rev,
        head_effects_fwd=head_fwd,
        head_effects_rev=head_rev,
        e_route_fwd=e_route,
        e_route_rev=m_TC - m_CC,
        joint_set_effects_fwd=_joint_for_topk(head_fwd, e_route, 10),
        joint_set_effects_rev=_joint_for_topk(head_rev, m_TC - m_CC, 10),
    )


def _expl_units():
    out = []
    for s in (42, 43, 44):
        u = _tiny_primary(s, name=f"exploratory_l1523_seed{s}", n=50)
        u.role = "exploratory_dev"
        u.family = "l1523"
        out.append(u)
    return out


def test_run_study_refuses_arbitrary_five_ids():
    expl = _expl_units()
    primaries = [
        _tiny_primary(s, name=f"arbitrary_seed{s}") for s in (42, 43, 44, 45, 46)
    ]
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["primary_roster_ok"] is False
    assert out["study_level"]["canonical_study_level_refused"] is True
    assert out["h3_study"]["H3a"] is None and out["h3_study"]["H3b"] is None
    assert out["h3_study"].get("roster_mismatch")


def test_run_study_refuses_renamed_primary():
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45)]
    primaries.append(_tiny_primary(46, name="primary_l19_seed99"))
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["primary_roster_ok"] is False
    assert "primary_l19_seed46" in out["primary_roster_mismatch"]["missing"]
    assert out["h3_study"]["H3a"] is None


def test_run_study_refuses_duplicate_primary_ids():
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45)]
    primaries.append(_tiny_primary(46, name="primary_l19_seed42"))  # duplicate 42
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["primary_roster_ok"] is False
    assert out["primary_roster_mismatch"]["duplicates"]
    assert out["h3_study"]["H3a"] is None


def test_run_study_refuses_missing_primary():
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45)]  # missing 46
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["primary_roster_ok"] is False
    assert "primary_l19_seed46" in out["primary_roster_mismatch"]["missing"]


def test_run_study_accepts_exact_frozen_roster():
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45, 46)]
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["primary_roster_ok"] is True
    assert out["primary_roster_mismatch"] is None
    assert "h3_direction_by_seed" in out
    assert len(out["h3_direction_by_seed"]) == 5
    # Canonical study_level is a string decision (not roster refusal)
    assert not (
        isinstance(out["study_level"], dict)
        and out["study_level"].get("roster_mismatch")
    )


def test_f1_preserved():
    assert f1_full_set_stream_sha256() == F1_REF
