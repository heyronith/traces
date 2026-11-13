"""Layer/type vs benign-damage matching are distinct."""

from __future__ import annotations

import pytest

from traces.pilot.components import build_full_l1523_catalog
from traces.pilot.controls import (
    DAMAGE_MATCH_ABS_TOLERANCE,
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


def test_damage_match_uses_validation_kl_only_and_can_unmatch():
    cat = build_full_l1523_catalog()
    target = [cat[0].component_id, cat[1].component_id, cat[2].component_id]
    cands = generate_layer_type_matched_candidates(cat, target, n_candidates=5, seed=9)
    # All candidates far from target K => unmatched
    cand_k = {c.name: 10.0 for c in cands}
    out = select_benign_damage_matched_control(
        target_ids=target,
        target_benign_kl=0.01,
        candidates=cands,
        candidate_benign_kl=cand_k,
        abs_tol=DAMAGE_MATCH_ABS_TOLERANCE,
        rel_tol=0.01,
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


def test_damage_match_selects_near_kl():
    cat = build_full_l1523_catalog()
    target = [cat[0].component_id, cat[8].component_id]
    cands = generate_layer_type_matched_candidates(cat, target, n_candidates=8, seed=3)
    cand_k = {c.name: 5.0 for c in cands}
    cand_k[cands[2].name] = 0.12  # near target 0.10
    out = select_benign_damage_matched_control(
        target_ids=target,
        target_benign_kl=0.10,
        candidates=cands,
        candidate_benign_kl=cand_k,
        abs_tol=0.05,
        rel_tol=0.5,
        split_used="validation",
    )
    assert out.kind == "benign_damage_matched"
    assert out.control_benign_kl == pytest.approx(0.12)
    assert out.target_benign_kl == pytest.approx(0.10)


def test_random_control():
    cat = build_full_l1523_catalog()
    r = random_control_set(cat, 5, seed=0)
    assert len(r.component_ids) == 5
    assert r.kind == "random"
