"""Agent 1 H3 follow-up: synthetic CPU only; scientific authorization OFF.

The supported H3a/H3b paths deliberately do NOT mock run_model_unit_analysis
or h3_study_level_from_seed_records. No model is instantiated or loaded.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from traces.q1.analysis_ext import h3_study_level_from_seed_records, selection_key
from traces.q1.component_sets import EXPECTED_UPSTREAM_HEADS
from traces.q1.driver import DriverResult, ModelUnitInput, PRIMARY_ROLE, run_study

ROSTER = [f"primary_l19_seed{s}" for s in range(42, 47)]
PAIR_A = frozenset({(15, 6), (16, 5)})
PAIR_B = frozenset({(14, 0), (14, 1)})
PAIR_C = frozenset({(13, 0), (13, 1)})
HELD = [f"held_{i:03d}" for i in range(200)]
DEV = [f"dev_{i:03d}" for i in range(50)]


def _qualifying_record(pair=PAIR_A):
    return {
        "full_h3_eligible": True,
        "h3_endpoint_status": "full_qualifying",
        "top2_for_h3": pair,
        "support_class": "full",
    }


def _five_records():
    return {sid: _qualifying_record() for sid in ROSTER}


@pytest.mark.parametrize("status,eligible", [
    ("full_qualifying", False),
    ("determinate_non_support", True),
])
def test_aggregator_blocks_original_contradictions(status, eligible):
    records = _five_records()
    records[ROSTER[0]] = {
        "full_h3_eligible": eligible,
        "h3_endpoint_status": status,
        "top2_for_h3": PAIR_A,
    }
    out = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert out["validation_blocked"] is True
    assert out["h3_endpoint_decision"] == "VALIDATION_BLOCKED"
    assert out["H3a"] is None and out["H3b"] is None
    assert out["endpoint_counts"]["technical_invalid"] == 1
    assert out["endpoint_counts"]["full_qualifying"] == 4


@pytest.mark.parametrize("bad_pair", [
    frozenset({(19, 0), (14, 0)}),
    frozenset({(-1, 0), (14, 0)}),
    frozenset({(True, 1), (14, 0)}),
])
def test_aggregator_refuses_bad_upstream_head_identity(bad_pair):
    records = _five_records()
    records[ROSTER[0]] = _qualifying_record(bad_pair)
    out = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert out["validation_blocked"] is True
    assert out["endpoint_counts"]["technical_invalid"] == 1
    assert out["H3a"] is None and out["H3b"] is None
    assert "non-upstream head IDs" in out["reason"]


def test_aggregator_refuses_unknown_explicit_status():
    records = _five_records()
    records[ROSTER[0]]["h3_endpoint_status"] = "R-weird"
    out = h3_study_level_from_seed_records(records, expected_roster=ROSTER)
    assert out["validation_blocked"] is True
    assert out["endpoint_counts"]["technical_invalid"] == 1


def _synthetic_joint_effects(head_fx, n):
    """Precomputed synthetic joint effects; no individual-head sum proxy."""
    ranked = sorted(head_fx, key=lambda h: (-float(head_fx[h][0]), h))
    return {
        selection_key(ranked[:k]): np.full(n, 0.7 if k == 1 else 2.05)
        for k in range(1, 11)
    }


def _measured_primary(seed, top2, ids=HELD, family="l19", role=PRIMARY_ROLE):
    n = len(ids)
    cc, tt = np.zeros(n), np.full(n, 3.0)
    ct, tc = np.full(n, 0.1), np.full(n, 2.9)
    e_fwd, e_rev = tt - ct, tc - cc

    # The complete 64-pair comparator makes the core share 0.05/2.9 < 0.25;
    # H1 fails decisively, independent of the nominated head identities.
    pairs = {(h15, h16): np.full(n, 0.20)
             for h15 in range(8) for h16 in range(8)}
    pairs[(6, 5)] = np.full(n, 0.05)

    heads = {h: np.full(n, 0.01) for h in EXPECTED_UPSTREAM_HEADS}
    for i, head in enumerate(sorted(top2)):
        heads[head] = np.full(n, 0.99 - 0.02 * i)
    joint_f = _synthetic_joint_effects(heads, n)
    joint_r = _synthetic_joint_effects(heads, n)

    return ModelUnitInput(
        model_unit=(f"primary_l19_seed{seed}" if role == PRIMARY_ROLE
                    else f"exploratory_l1523_seed{seed}"),
        family=family,
        role=role,
        prompt_ids=list(ids),
        m_TT=tt, m_CT=ct, m_TC=tc, m_CC=cc,
        eligible_mask=np.ones(n, dtype=bool),
        trigger_asr=0.99,
        clean_false_fire=0.01,
        first_token_validity=0.99,
        n_frozen=n,
        pair_effects_fwd=pairs,
        pair_effects_rev=pairs,
        head_effects_fwd=heads,
        head_effects_rev=heads,
        e_route_fwd=e_fwd,
        e_route_rev=e_rev,
        joint_set_effects_fwd=joint_f,
        joint_set_effects_rev=joint_r,
    )


def _exploratory_roster():
    return [
        _measured_primary(s, PAIR_A, DEV, family="l1523", role="exploratory_dev")
        for s in (42, 43, 44)
    ]


@pytest.mark.parametrize("pairs,hypothesis", [
    ([PAIR_A, PAIR_A, PAIR_A, PAIR_A, PAIR_B], "H3a"),
    ([PAIR_A, PAIR_A, PAIR_A, PAIR_B, PAIR_C], "H3b"),
])
def test_real_per_unit_run_study_h3_supported_strict_json(pairs, hypothesis):
    # Actual route, H1, 10-step F2 crossfit, per-seed H3 and study aggregator.
    # No patched run_model_unit_analysis or h3_study_level_from_seed_records.
    primaries = [_measured_primary(seed, pair)
                 for seed, pair in zip(range(42, 47), pairs)]
    out = run_study(
        _exploratory_roster() + primaries, HELD,
        exploratory_single_bos_ranks=[1, 2, 3],
        development_ids_allocation_order=DEV,
    )
    assert out["primary_roster_ok"] is True
    assert out["allocation"]["held_out_ok"] is True
    assert out["h3_study"][hypothesis] == "supported"
    other = "H3b" if hypothesis == "H3a" else "H3a"
    assert out["h3_study"][other] is None
    assert out["h3_study"]["n_qualifying"] == 5
    assert out["h3_study"]["validation_blocked"] is False
    assert len(out["h3_top2_by_seed"]) == 5
    assert len(out["h3_direction_by_seed"]) == 5
    assert out["study_level"] is not None
    for row in out["per_unit"]:
        if row.get("role") != PRIMARY_ROLE:
            continue
        assert row["route"]["route_class"] == "R-moved"
        assert row["h1"]["full_H1"] == "H1-fail"
        assert row["h2_h3"]["h3_unit"]["h3_endpoint_status"] == "full_qualifying"
        assert row["h2_h3"]["h3_unit"]["k_star_band_fwd"] == "le_2"
        assert row["h2_h3"]["h3_unit"]["k_star_band_rev"] == "le_2"
    json.dumps(out, allow_nan=False)  # strict; NO default=str


@pytest.mark.parametrize("status,eligible", [
    ("full_qualifying", False),
    ("determinate_non_support", True),
])
def test_run_study_does_not_erase_contradiction(monkeypatch, status, eligible):
    import traces.q1.driver as drv

    original = drv.run_model_unit_analysis

    def inject(unit, held, *, top2_by_seed=None):
        if unit.role != PRIMARY_ROLE:
            return original(unit, held, top2_by_seed=top2_by_seed)
        raw = _qualifying_record()
        if unit.model_unit == ROSTER[0]:
            raw["full_h3_eligible"] = eligible
            raw["h3_endpoint_status"] = status
        return DriverResult(
            model_unit=unit.model_unit,
            gates={"passed": True, "reasons": []},
            route={"route_class": "R-moved"},
            h1={"full_H1": "H1-fail"},
            h2_h3={"H2": None, "H3a": None, "H3b": None, "h3_unit": raw},
            crossfit={},
            meta={},
        )

    monkeypatch.setattr(drv, "run_model_unit_analysis", inject)
    primaries = [_measured_primary(s, PAIR_A) for s in range(42, 47)]
    out = run_study(
        _exploratory_roster() + primaries, HELD,
        exploratory_single_bos_ranks=[1, 2, 3],
        development_ids_allocation_order=DEV,
    )
    assert out["h3_direction_by_seed"][ROSTER[0]]["full_h3_eligible"] is eligible
    assert out["h3_study"]["validation_blocked"] is True
    assert out["h3_study"]["h3_endpoint_decision"] == "VALIDATION_BLOCKED"
    assert out["h3_study"]["H3a"] is None and out["h3_study"]["H3b"] is None
    json.dumps(out, allow_nan=False)
