"""H3 second re-audit targeted corrections — synthetic CPU, no forwards.

Owner-approved decisive-over-uncertain H3-B; family-bound H3-C; two real
``run_study`` end-to-end H3a/H3b success paths. Scientific authorization OFF.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from traces.q1.analysis_ext import (
    FROZEN_H3_ROUTE_CLASSES,
    assess_unit_h3_direction_support,
    h3_study_level_from_seed_records,
)
from traces.q1.component_sets import EXPECTED_UPSTREAM_HEADS
from traces.q1.driver import (
    DriverResult,
    ModelUnitInput,
    PRIMARY_ROLE,
    run_study,
)
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256

F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"
PAIR_A = frozenset([(15, 6), (16, 5)])
PAIR_B = frozenset([(14, 0), (14, 1)])
PAIR_C = frozenset([(13, 0), (13, 1)])
ROSTER = [f"primary_l19_seed{s}" for s in (42, 43, 44, 45, 46)]


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


def _assess(**kw):
    defaults = dict(
        denom_fwd_ok=True,
        denom_rev_ok=True,
        head_effects_fwd=_effects(PAIR_A),
        head_effects_rev=_effects(PAIR_A),
        crossfit_fwd={"k_star": _k("le_2")},
        crossfit_rev={"k_star": _k("le_2")},
        n_eligible=8,
        gates_passed=True,
        h1_full_state="H1-fail",
        route_class="R-moved",
    )
    defaults.update(kw)
    return assess_unit_h3_direction_support(**defaults)


# --- Must-fix 1: decisive-over-uncertain ------------------------------------


def test_probe_a_rmoved_disagree_fwd_undetermined_is_determinate_non_support():
    out = _assess(
        head_effects_fwd=_effects(PAIR_A),
        head_effects_rev=_effects(PAIR_B),
        crossfit_fwd={"k_star": _k("undetermined")},
        crossfit_rev={"k_star": _k("le_2")},
    )
    assert out["h3_endpoint_status"] == "determinate_non_support"
    assert out["full_h3_eligible"] is False
    assert out["preserved_uncertainty"] is not None
    assert out["preserved_uncertainty"]["k_star_band_fwd"] == "undetermined"
    assert out["rmoved_top2_agree"] is False


def test_probe_b_disagree_plus_h1_indeterminate_is_determinate_non_support():
    out = _assess(
        h1_full_state="H1-indeterminate",
        head_effects_fwd=_effects(PAIR_A),
        head_effects_rev=_effects(PAIR_B),
    )
    assert out["h3_endpoint_status"] == "determinate_non_support"
    assert out["preserved_uncertainty"]["h1_state"] == "H1-indeterminate"


def test_probe_c_rev_gt10_fwd_undetermined_is_determinate_non_support():
    out = _assess(
        head_effects_fwd=_effects(PAIR_A),
        head_effects_rev=_effects(PAIR_A),
        crossfit_fwd={"k_star": _k("undetermined")},
        crossfit_rev={"k_star": _k("gt_10")},
    )
    assert out["h3_endpoint_status"] == "determinate_non_support"
    assert out["support_class"] == "k_star_gt_10"
    assert out["preserved_uncertainty"]["k_star_band_fwd"] == "undetermined"


def test_probe_d_incomplete_maps_cannot_claim_disagreement():
    incomplete = {h: np.full(8, 0.01) for h in list(EXPECTED_UPSTREAM_HEADS)[:-1]}
    out = _assess(
        head_effects_fwd=incomplete,
        head_effects_rev=_effects(PAIR_B),
        h1_full_state="H1-indeterminate",
    )
    assert out["h3_endpoint_status"] == "h3_indeterminate"
    assert out["support_class"] == "incomplete_ranking"
    assert out["rmoved_top2_agree"] is None


def test_probe_e_no_disqualifier_h1_indeterminate_remains_indeterminate():
    out = _assess(h1_full_state="H1-indeterminate")
    assert out["h3_endpoint_status"] == "h3_indeterminate"
    assert out["uncertainty_kind"] == "h1_indeterminate"


def test_probe_f_gate_failed_unchanged():
    out = _assess(gates_passed=False)
    assert out["h3_endpoint_status"] == "gate_failed"


def test_probe_g_matching_rmoved_both_le2_h1_fail_qualifies():
    out = _assess()
    assert out["h3_endpoint_status"] == "full_qualifying"
    assert out["full_h3_eligible"] is True
    assert out["top2_for_h3"] == PAIR_A


# --- Must-fix 2: unknown routes fail closed ---------------------------------


def test_unknown_route_raises_not_determinate_non_support():
    with pytest.raises(ValueError, match="unknown route_class"):
        _assess(route_class="R-weird")


def test_frozen_route_vocabulary():
    assert FROZEN_H3_ROUTE_CLASSES == frozenset(
        {"R-moved", "R-and", "R-or", "R-tag", "R-intermediate", "Indeterminate"}
    )


def test_legitimate_indeterminate_route_is_endpoint_indeterminate():
    out = _assess(route_class="Indeterminate")
    assert out["h3_endpoint_status"] == "h3_indeterminate"


# --- Must-fix 3: contradictory records -------------------------------------


def test_contradictory_full_qualifying_ineligible_validation_blocked():
    records = {sid: {
        "full_h3_eligible": True,
        "top2_for_h3": PAIR_A,
        "h3_endpoint_status": "full_qualifying",
        "support_class": "full",
    } for sid in ROSTER}
    records[ROSTER[0]] = {
        "full_h3_eligible": False,
        "top2_for_h3": PAIR_A,
        "h3_endpoint_status": "full_qualifying",
        "support_class": "full",
        "reason": "forged",
    }
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["validation_blocked"] is True
    assert study["H3a"] is None and study["H3b"] is None
    assert study["h3_endpoint_decision"] == "VALIDATION_BLOCKED"
    assert study["endpoint_counts"].get("full_qualifying", 0) == 4
    assert study["endpoint_counts"].get("technical_invalid", 0) >= 1
    json.dumps(study, allow_nan=False, default=str)


def test_contradictory_eligible_non_qualifying_status_blocked():
    records = {sid: {
        "full_h3_eligible": True,
        "top2_for_h3": PAIR_A,
        "h3_endpoint_status": "full_qualifying",
        "support_class": "full",
    } for sid in ROSTER}
    records[ROSTER[1]] = {
        "full_h3_eligible": True,
        "top2_for_h3": PAIR_A,
        "h3_endpoint_status": "determinate_non_support",
        "support_class": "route_inapplicable",
    }
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["validation_blocked"] is True
    assert study["H3a"] is None and study["H3b"] is None


def test_missing_top2_on_qualifying_blocked():
    records = {sid: {
        "full_h3_eligible": True,
        "top2_for_h3": PAIR_A,
        "h3_endpoint_status": "full_qualifying",
        "support_class": "full",
    } for sid in ROSTER}
    records[ROSTER[2]] = {
        "full_h3_eligible": True,
        "top2_for_h3": None,
        "h3_endpoint_status": "full_qualifying",
        "support_class": "full",
    }
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["validation_blocked"] is True
    assert study["H3a"] is None


# --- Fixtures for run_study integration -------------------------------------


def _joint_for_topk(head_fx: dict, e_route: np.ndarray, k: int) -> dict:
    means = {h: float(np.mean(v)) for h, v in head_fx.items()}
    ranked = sorted(means.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    out = {}
    for kk in range(1, k + 1):
        sel = [h for h, _ in ranked[:kk]]
        from traces.q1.analysis_ext import selection_key

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


def _qualifying_h3_unit(top2: frozenset) -> dict[str, Any]:
    return {
        "full_h3_eligible": True,
        "top2_for_h3": sorted(top2),
        "top2_forward": sorted(top2),
        "top2_reverse": sorted(top2),
        "applicable_directions": ["forward", "reverse"],
        "route_class": "R-moved",
        "h1_state": "H1-fail",
        "gates_passed": True,
        "support_class": "full",
        "h3_endpoint_status": "full_qualifying",
        "reason": None,
        "uncertainty_kind": None,
        "k_star_le_2_fwd": True,
        "k_star_le_2_rev": True,
        "k_star_band_fwd": "le_2",
        "k_star_band_rev": "le_2",
        "rmoved_top2_agree": True,
    }


def _indeterminate_h3_unit() -> dict[str, Any]:
    return {
        "full_h3_eligible": False,
        "top2_for_h3": None,
        "top2_forward": sorted(PAIR_A),
        "top2_reverse": sorted(PAIR_A),
        "applicable_directions": ["forward"],
        "route_class": "R-and",
        "h1_state": "H1-indeterminate",
        "gates_passed": True,
        "support_class": "h1_not_fail",
        "h3_endpoint_status": "h3_indeterminate",
        "reason": "H1-indeterminate on applicable route — H3-indeterminate",
        "uncertainty_kind": "h1_indeterminate",
        "k_star_le_2_fwd": True,
        "k_star_le_2_rev": True,
        "k_star_band_fwd": "le_2",
        "k_star_band_rev": "le_2",
        "rmoved_top2_agree": None,
    }


def _stub_primary_result(unit: ModelUnitInput, h3_unit: dict[str, Any]) -> DriverResult:
    return DriverResult(
        model_unit=unit.model_unit,
        gates={"passed": True, "reasons": []},
        route={"route_class": h3_unit.get("route_class", "R-moved")},
        h1={"full_H1": h3_unit.get("h1_state", "H1-fail")},
        h2_h3={
            "H2": None,
            "H3a": None,
            "H3b": None,
            "h3_unit": dict(h3_unit),
        },
        crossfit={"indeterminate": False},
        meta={},
    )


def _patch_primary_h3(monkeypatch, by_id: dict[str, dict[str, Any]]):
    """Scoped patch of per-unit analysis only — run_study / aggregator stay real."""
    import traces.q1.driver as drv

    real = drv.run_model_unit_analysis

    def _fake(unit, held, *, top2_by_seed=None):
        if unit.role != PRIMARY_ROLE:
            return real(unit, held, top2_by_seed=top2_by_seed)
        h3u = by_id[unit.model_unit]
        return _stub_primary_result(unit, h3u)

    monkeypatch.setattr(drv, "run_model_unit_analysis", _fake)


# --- Must-fix 4: family binding --------------------------------------------


def test_run_study_refuses_family_spoof_l1523():
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45, 46)]
    primaries[2].family = "l1523"  # correct ID, wrong family
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["primary_roster_ok"] is False
    mm = out["primary_roster_mismatch"]
    assert mm["family_failures"]
    assert any(f["model_unit"] == "primary_l19_seed44" for f in mm["family_failures"])
    assert out["h3_study"]["H3a"] is None and out["h3_study"]["H3b"] is None
    assert out["study_level"]["canonical_study_level_refused"] is True
    json.dumps(out, allow_nan=False, default=str)


def test_run_study_refuses_family_spoof_base():
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45, 46)]
    primaries[0].family = "base"
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["primary_roster_ok"] is False
    assert out["primary_roster_mismatch"]["family_failures"][0]["family"] == "base"
    assert out["h3_study"]["H3a"] is None


# --- Must-fix 5: real run_study H3a / H3b e2e -------------------------------


def test_run_study_e2e_h3a_supported(monkeypatch):
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45, 46)]
    by_id = {u.model_unit: _qualifying_h3_unit(PAIR_A) for u in primaries}
    by_id[ROSTER[4]] = _qualifying_h3_unit(PAIR_B)  # 4+1 → H3a only
    _patch_primary_h3(monkeypatch, by_id)
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])

    assert out["primary_roster_ok"] is True
    h3 = out["h3_study"]
    assert h3["H3a"] == "supported"
    assert h3["H3b"] is None
    assert h3["H3b_eligible"] is False
    assert h3["H3a_count"] == 4
    assert h3["n_qualifying"] == 5
    assert h3["h3_endpoint_decision"] == "H3a_SUPPORTED"
    assert h3.get("validation_blocked") is False
    assert len(out["h3_direction_by_seed"]) == 5
    assert len(out["h3_top2_by_seed"]) == 5
    assert out["study_level"] is not None
    assert not (
        isinstance(out["study_level"], dict)
        and out["study_level"].get("roster_mismatch")
    )
    json.dumps(out, allow_nan=False, default=str)


def test_run_study_e2e_h3b_supported(monkeypatch):
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45, 46)]
    pairs = [PAIR_A, PAIR_A, PAIR_A, PAIR_B, PAIR_C]  # 3+1+1 among 5 → H3b
    by_id = {
        u.model_unit: _qualifying_h3_unit(p) for u, p in zip(primaries, pairs)
    }
    _patch_primary_h3(monkeypatch, by_id)
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])

    assert out["primary_roster_ok"] is True
    h3 = out["h3_study"]
    assert h3["H3b"] == "supported"
    assert h3["H3a"] is None
    assert h3["H3a_eligible"] is False
    assert h3["n_qualifying"] == 5
    assert h3["H3a_count"] == 3
    assert h3["h3_endpoint_decision"] == "H3b_SUPPORTED"
    assert len(out["h3_top2_by_seed"]) == 5
    json.dumps(out, allow_nan=False, default=str)


def test_run_study_e2e_h3_endpoint_inconclusive_precedence(monkeypatch):
    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45, 46)]
    by_id = {u.model_unit: _qualifying_h3_unit(PAIR_A) for u in primaries}
    by_id[ROSTER[0]] = _indeterminate_h3_unit()
    by_id[ROSTER[1]] = _indeterminate_h3_unit()
    _patch_primary_h3(monkeypatch, by_id)
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    h3 = out["h3_study"]
    assert h3["h3_endpoint_inconclusive"] is True
    assert h3["H3a"] is None and h3["H3b"] is None
    assert h3["h3_endpoint_decision"] == "INCONCLUSIVE"
    json.dumps(out, allow_nan=False, default=str)


def test_run_study_unknown_route_surfaces_technical_invalid(monkeypatch):
    """Unknown route via assess ValueError must not rewrite to determinate_non_support."""
    import traces.q1.driver as drv

    expl = _expl_units()
    primaries = [_tiny_primary(s) for s in (42, 43, 44, 45, 46)]
    real = drv.run_model_unit_analysis

    def _fake(unit, held, *, top2_by_seed=None):
        if unit.role != PRIMARY_ROLE:
            return real(unit, held, top2_by_seed=top2_by_seed)
        if unit.model_unit == ROSTER[0]:
            return DriverResult(
                model_unit=unit.model_unit,
                gates={"passed": True, "reasons": []},
                route={"route_class": "R-weird"},
                h1={"full_H1": "H1-fail"},
                h2_h3={
                    "H2": None,
                    "H3a": None,
                    "H3b": None,
                    "h3_unit": {
                        "full_h3_eligible": False,
                        "top2_for_h3": None,
                        "route_class": "R-weird",
                        "support_class": "technical_invalid",
                        "h3_endpoint_status": "technical_invalid",
                        "reason": "unknown route_class 'R-weird'",
                        "gates_passed": True,
                    },
                },
                crossfit={},
                meta={},
            )
        return _stub_primary_result(unit, _qualifying_h3_unit(PAIR_A))

    monkeypatch.setattr(drv, "run_model_unit_analysis", _fake)
    held = primaries[0].prompt_ids + [f"h{i}" for i in range(180)]
    out = run_study(expl + primaries, held, exploratory_single_bos_ranks=[1, 2, 3])
    meta0 = out["h3_direction_by_seed"][ROSTER[0]]
    assert meta0["h3_endpoint_status"] == "technical_invalid"
    assert meta0["h3_endpoint_status"] != "determinate_non_support"
    h3 = out["h3_study"]
    assert h3["H3a"] is None and h3["H3b"] is None
    assert h3.get("validation_blocked") is True or h3.get("h3_endpoint_decision") in {
        "VALIDATION_BLOCKED",
        "INCONCLUSIVE",
        "NOT_SUPPORTED",
    }
    json.dumps(out, allow_nan=False, default=str)


def test_h3a_mutual_exclusion_preserved():
    records = {sid: {
        "full_h3_eligible": True,
        "top2_for_h3": PAIR_A,
        "h3_endpoint_status": "full_qualifying",
        "support_class": "full",
    } for sid in ROSTER[:4]}
    records[ROSTER[4]] = {
        "full_h3_eligible": True,
        "top2_for_h3": PAIR_B,
        "h3_endpoint_status": "full_qualifying",
        "support_class": "full",
    }
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["H3a"] == "supported"
    assert study["H3b"] is None
    assert study["H3b_eligible"] is False


def test_h3b_3plus2_preserved():
    records = {
        ROSTER[0]: {
            "full_h3_eligible": True,
            "top2_for_h3": PAIR_A,
            "h3_endpoint_status": "full_qualifying",
            "support_class": "full",
        },
        ROSTER[1]: {
            "full_h3_eligible": True,
            "top2_for_h3": PAIR_A,
            "h3_endpoint_status": "full_qualifying",
            "support_class": "full",
        },
        ROSTER[2]: {
            "full_h3_eligible": True,
            "top2_for_h3": PAIR_A,
            "h3_endpoint_status": "full_qualifying",
            "support_class": "full",
        },
        ROSTER[3]: {
            "full_h3_eligible": True,
            "top2_for_h3": PAIR_B,
            "h3_endpoint_status": "full_qualifying",
            "support_class": "full",
        },
        ROSTER[4]: {
            "full_h3_eligible": True,
            "top2_for_h3": PAIR_B,
            "h3_endpoint_status": "full_qualifying",
            "support_class": "full",
        },
    }
    study = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert study["H3b"] == "supported"
    assert study["H3a"] is None


def test_f1_preserved():
    assert f1_full_set_stream_sha256() == F1_REF
