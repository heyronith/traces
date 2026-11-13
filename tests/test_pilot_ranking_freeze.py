"""Donor rankings freeze before recipient eval; leakage/mutation fail."""

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
    TransferIntegrityContext,
    build_transfer_conditions,
    validate_frozen_ranking_for_transfer,
)


def _rank(scores, seed=42, catalog=None):
    cat = catalog or build_full_l1523_catalog()
    # Pad scores for all components minimally
    full = {c.component_id: scores.get(c.component_id, 0.0) for c in cat}
    full.update(scores)
    return rank_components_from_scores(
        full,
        donor_seed=seed,
        ranking_split="discovery",
        n_prompts=50,
        catalog=cat,
        prompt_set_hash="ph",
        dataset_split_hash="dh",
    )


def test_full_catalog_is_396():
    cat = build_full_l1523_catalog()
    assert len(cat) == 396
    summary = catalog_summary(cat)
    assert summary["n_attention_heads"] == 72
    assert summary["n_mlp_groups"] == 324


def test_ranking_cannot_use_held_out_or_validation():
    cat = build_full_l1523_catalog()
    with pytest.raises(RuntimeError):
        rank_components_from_scores(
            {cat[0].component_id: 1.0},
            donor_seed=42,
            ranking_split="held_out",
            n_prompts=10,
            catalog=cat,
            prompt_set_hash="p",
            dataset_split_hash="d",
        )
    with pytest.raises(RuntimeError):
        rank_components_from_scores(
            {cat[0].component_id: 1.0},
            donor_seed=42,
            ranking_split="validation",
            n_prompts=10,
            catalog=cat,
            prompt_set_hash="p",
            dataset_split_hash="d",
        )


def test_freeze_save_load_immutable(tmp_path: Path):
    cat = build_full_l1523_catalog()
    scores = {c.component_id: float(i % 17) for i, c in enumerate(cat)}
    ranking = _rank(scores, catalog=cat)
    path = tmp_path / "rank42.json"
    h = save_frozen_ranking(ranking, path)
    loaded = load_frozen_ranking(path)
    assert loaded.frozen is True
    assert loaded.artifact_hash() == h
    assert isinstance(loaded.component_order, tuple)
    with pytest.raises(TypeError):
        loaded.component_order[0] = "mutated"  # type: ignore[index]


def test_tampered_artifact_rejected(tmp_path: Path):
    ranking = _rank({"L15.attn.head0": 2.0, "L15.attn.head1": 1.0})
    path = tmp_path / "rank.json"
    save_frozen_ranking(ranking, path)
    text = path.read_text().replace('"score": 2.0', '"score": 9.0', 1)
    path.write_text(text)
    with pytest.raises(RuntimeError, match="artifact_hash"):
        load_frozen_ranking(path)


def test_transfer_integrity_checks(tmp_path: Path):
    cat = build_full_l1523_catalog()
    donor = _rank({cat[0].component_id: 3.0, cat[1].component_id: 2.0}, seed=42, catalog=cat)
    save_frozen_ranking(donor, tmp_path / "d.json")
    frozen = load_frozen_ranking(tmp_path / "d.json")
    ctx = TransferIntegrityContext(
        donor_seed=42,
        recipient_seed=43,
        catalog_hash=frozen.catalog_hash,
        dataset_split_hash=frozen.dataset_split_hash,
    )
    validate_frozen_ranking_for_transfer(
        frozen, catalog=cat, ctx=ctx, expected_donor_seed=42
    )
    # Wrong catalog hash fails
    bad = TransferIntegrityContext(
        donor_seed=42,
        recipient_seed=43,
        catalog_hash="nope",
        dataset_split_hash=frozen.dataset_split_hash,
    )
    with pytest.raises(RuntimeError, match="catalog"):
        validate_frozen_ranking_for_transfer(
            frozen, catalog=cat, ctx=bad, expected_donor_seed=42
        )


def test_invalid_k_and_duplicate_components_fail():
    cat = build_full_l1523_catalog()
    donor = _rank({c.component_id: 1.0 for c in cat[:20]}, seed=42, catalog=cat)
    with pytest.raises(ValueError):
        donor.top_k(0)
    with pytest.raises(ValueError):
        donor.top_k(10_000)
    recip = _rank({c.component_id: 0.5 for c in cat[:20]}, seed=43, catalog=cat)
    ctx = TransferIntegrityContext(
        donor_seed=42,
        recipient_seed=43,
        catalog_hash=donor.catalog_hash,
        dataset_split_hash=donor.dataset_split_hash,
    )
    top = list(donor.top_k(2))
    rnd = random_control_set(cat, 2, seed=1, exclude=top)
    lt = layer_type_matched_control(cat, top, seed=2)
    dup = [rnd.component_ids[0], rnd.component_ids[0]]
    with pytest.raises(RuntimeError, match="Duplicated"):
        build_transfer_conditions(
            recipient_seed=43,
            donor_ranking=donor,
            recipient_native_ranking=recip,
            k=2,
            random_ids=dup,
            layer_type_ids=list(lt.component_ids),
            damage_matched_ids=None,
            damage_unmatched=True,
            catalog=cat,
            ctx=ctx,
        )


def test_discovery_folds_disjoint():
    ids = [f"pilot_{i:03d}" for i in range(50)]
    folds = split_discovery_for_reliability(ids, seed=7, n_folds=2)
    assert set(folds[0]).isdisjoint(set(folds[1]))
