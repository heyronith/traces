"""Layer/type vs benign-damage matching are distinct; v2.1 bound is conservative."""

from __future__ import annotations

import pytest

from traces.pilot.components import build_full_l1523_catalog
from traces.pilot.controls import (
    damage_match_permitted_bound,
    generate_layer_type_matched_candidates,
    layer_type_matched_control,
    random_control_set,
    select_benign_damage_matched_control,
)


def test_layer_type_and_damage_are_different_kinds():
    cat = build_full_l1523_catalog()
    target = [c.component_id for c in cat[:5] if c.kind == "attention_head"][:3]
    lt = layer_type_matched_control(cat, target, seed=1)
    assert lt.kind == "layer_type_matched"
    assert "NOT selected by benign KL" in lt.notes or "NOT" in lt.notes


def test_v21_bound_rejects_small_kl_far_control():
    """K_target=0.005 and K_control=0.050 must not match under v2.1 bound."""
    k_t = 0.005
    k_c = 0.050
    bound = damage_match_permitted_bound(k_t)
    # min(0.005, 0.25 * max(0.005, 0.002)) = min(0.005, 0.00125) = 0.00125
    assert bound == pytest.approx(0.00125)
    assert abs(k_c - k_t) > bound

    cat = build_full_l1523_catalog()
    target = [cat[0].component_id, cat[1].component_id]
    cands = generate_layer_type_matched_candidates(cat, target, n_candidates=3, seed=1)
    cand_k = {c.name: k_c for c in cands}
    out = select_benign_damage_matched_control(
        target_ids=target,
        target_benign_kl=k_t,
        candidates=cands,
        candidate_benign_kl=cand_k,
        split_used="validation",
    )
    assert out.kind == "unmatched"
    assert out.match_permitted_bound == pytest.approx(bound)


def test_damage_match_uses_validation_kl_only_and_can_unmatch():
    cat = build_full_l1523_catalog()
    target = [cat[0].component_id, cat[1].component_id, cat[2].component_id]
    cands = generate_layer_type_matched_candidates(cat, target, n_candidates=5, seed=9)
    cand_k = {c.name: 10.0 for c in cands}
    out = select_benign_damage_matched_control(
        target_ids=target,
        target_benign_kl=0.01,
        candidates=cands,
        candidate_benign_kl=cand_k,
        split_used="validation",
    )
    assert out.kind == "unmatched"
    assert out.unmatched_reason is not None


def test_damage_match_forbids_held_out():
    with pytest.raises(RuntimeError, match="held_out"):
        select_benign_damage_matched_control(
            target_ids=["a"],
            target_benign_kl=0.1,
            candidates=[],
            candidate_benign_kl={},
            split_used="held_out",
        )


def test_damage_match_selects_within_v21_bound():
    cat = build_full_l1523_catalog()
    target = [cat[0].component_id, cat[8].component_id]
    cands = generate_layer_type_matched_candidates(cat, target, n_candidates=8, seed=3)
    k_t = 0.10
    bound = damage_match_permitted_bound(k_t)  # min(0.005, 0.025) = 0.005
    cand_k = {c.name: 5.0 for c in cands}
    cand_k[cands[2].name] = k_t + 0.004  # within 0.005
    out = select_benign_damage_matched_control(
        target_ids=target,
        target_benign_kl=k_t,
        candidates=cands,
        candidate_benign_kl=cand_k,
        split_used="validation",
    )
    assert out.kind == "benign_damage_matched"
    assert out.control_benign_kl == pytest.approx(k_t + 0.004)
    assert out.match_abs_diff == pytest.approx(0.004)
    assert out.match_permitted_bound == pytest.approx(bound)


def test_legacy_or_tolerance_would_have_matched_but_v21_does_not():
    """Regression: old abs_tol=0.05 OR rel_tol=0.25 would accept 0.005 vs 0.050."""
    # relative err = 0.045/0.005 = 9 >> 0.25; absolute 0.045 < 0.05 — old OR would match
    assert abs(0.050 - 0.005) <= 0.05  # old abs would pass
    assert abs(0.050 - 0.005) > damage_match_permitted_bound(0.005)


def test_random_control():
    cat = build_full_l1523_catalog()
    r = random_control_set(cat, 5, seed=0)
    assert len(r.component_ids) == 5
    assert r.kind == "random"
