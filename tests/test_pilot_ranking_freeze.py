"""Donor rankings freeze before recipient eval; no held_out leakage."""

from __future__ import annotations

from pathlib import Path

import pytest

from traces.pilot.components import build_full_l1523_catalog, catalog_summary
from traces.pilot.controls import layer_type_matched_control, random_control_set
from traces.pilot.ranking import (
    load_frozen_ranking,
    rank_components_from_scores,
    save_frozen_ranking,
    split_discovery_for_reliability,
)
from traces.pilot.transfer import (
    assert_donor_frozen_before_recipient,
    build_transfer_conditions,
)


def test_full_catalog_is_396():
    cat = build_full_l1523_catalog()
    assert len(cat) == 396
    summary = catalog_summary(cat)
    assert summary["n_attention_heads"] == 72
    assert summary["n_mlp_groups"] == 324


def test_ranking_cannot_use_held_out():
    with pytest.raises(RuntimeError):
        rank_components_from_scores(
            {"L15.attn.head0": 1.0},
            donor_seed=42,
            ranking_split="held_out",
            n_prompts=10,
        )


def test_freeze_save_load_and_hash(tmp_path: Path):
    cat = build_full_l1523_catalog()
    scores = {c.component_id: float(i % 17) for i, c in enumerate(cat)}
    ranking = rank_components_from_scores(
        scores,
        donor_seed=42,
        ranking_split="discovery",
        n_prompts=50,
        catalog=cat,
        prompt_set_hash="abc",
    )
    path = tmp_path / "rank42.json"
    h = save_frozen_ranking(ranking, path)
    loaded = load_frozen_ranking(path)
    assert loaded.frozen is True
    assert loaded.donor_seed == 42
    assert loaded.artifact_hash() == h
    assert loaded.top_k(5) == ranking.top_k(5)


def test_tampered_artifact_rejected(tmp_path: Path):
    ranking = rank_components_from_scores(
        {"a": 2.0, "b": 1.0},
        donor_seed=43,
        ranking_split="discovery",
        n_prompts=10,
    )
    path = tmp_path / "rank.json"
    save_frozen_ranking(ranking, path)
    text = path.read_text().replace('"score": 2.0', '"score": 9.0', 1)
    path.write_text(text)
    with pytest.raises(RuntimeError, match="artifact_hash"):
        load_frozen_ranking(path)


def test_donor_frozen_before_recipient_eval(tmp_path: Path):
    ranking = rank_components_from_scores(
        {"L15.attn.head0": 3.0, "L16.mlp.g0": 2.0, "L17.attn.head1": 1.0},
        donor_seed=42,
        ranking_split="discovery",
        n_prompts=50,
    )
    save_frozen_ranking(ranking, tmp_path / "d.json")
    frozen = load_frozen_ranking(tmp_path / "d.json")
    assert_donor_frozen_before_recipient(frozen, recipient_eval_split="held_out")
    # Recipient conditions must use frozen top-k, not re-rank
    cat = build_full_l1523_catalog()
    top = frozen.top_k(2)
    rnd = random_control_set(cat, 2, seed=1, exclude=top)
    matched = layer_type_matched_control(cat, top, seed=2)
    native = rank_components_from_scores(
        {c.component_id: 0.1 for c in cat[:20]},
        donor_seed=43,
        ranking_split="discovery",
        n_prompts=50,
        catalog=cat,
    )
    conds = build_transfer_conditions(
        recipient_seed=43,
        donor_ranking=frozen,
        recipient_native_ranking=native,
        k=2,
        random_ids=rnd.component_ids,
        matched_ids=matched.component_ids,
    )
    transfer = next(c for c in conds if c.source == "frozen_donor_topk")
    assert transfer.component_ids == top


def test_discovery_folds_disjoint():
    ids = [f"pilot_{i:03d}" for i in range(50)]
    folds = split_discovery_for_reliability(ids, seed=7, n_folds=2)
    assert set(folds[0]).isdisjoint(set(folds[1]))
    assert sorted(folds[0] + folds[1]) == sorted(ids)
