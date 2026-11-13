"""CPU synthetic end-to-end pilot workflow (no GPU)."""

from __future__ import annotations

import pytest

from traces.pilot.engine import run_pilot_pipeline
from traces.pilot.runner import assert_pilot_disabled_or_authorized, load_pilot_config
from traces.pilot.synthetic import (
    SyntheticPilotScorer,
    make_disjoint_causal_world,
    make_shared_causal_world,
)


def test_config_blocks_real_execution_path():
    cfg = load_pilot_config()
    assert cfg.get("enabled") is False
    assert_pilot_disabled_or_authorized(cfg)
    with pytest.raises(RuntimeError, match="blocked|disabled|authorized|permitted"):
        run_pilot_pipeline(SyntheticPilotScorer(make_shared_causal_world()), allow_execution=False)


def test_e2e_shared_causal_recovers_transfer(tmp_path):
    world = make_shared_causal_world(k_plant=10)
    scorer = SyntheticPilotScorer(world)
    result = run_pilot_pipeline(
        scorer,
        seeds=(42, 43, 44),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=tmp_path,
        allow_execution=True,
    )
    assert result.executed is True
    # Rankings should put planted components near the top
    for seed in (42, 43, 44):
        top = set(result.per_seed[str(seed)]["top10"])
        planted = set(world.causal_by_seed[seed][:10])
        assert len(top & planted) >= 7

    # Transfer D should be high when donor/recipient share causal set
    transfer_rows = [
        r
        for r in result.per_condition
        if r["condition"]["source"] == "frozen_donor_topk" and r["split"] == "held_out"
    ]
    assert transfer_rows
    assert sum(r["D"] or 0 for r in transfer_rows) / len(transfer_rows) > 0.4

    # Artifact hashes present
    assert all(
        r["condition"].get("donor_ranking_artifact_hash")
        for r in transfer_rows
    )


def test_e2e_disjoint_causal_transfer_weaker_than_native(tmp_path):
    world = make_disjoint_causal_world(k_plant=10)
    scorer = SyntheticPilotScorer(world)
    result = run_pilot_pipeline(
        scorer,
        seeds=(42, 43, 44),
        k_budgets=(10,),
        catalog=world.catalog,
        output_dir=tmp_path,
        allow_execution=True,
    )
    rows = [
        r
        for r in result.per_condition
        if r["split"] == "held_out" and r["condition"]["k"] == 10
    ]
    transfer = next(
        r
        for r in rows
        if r["condition"]["source"] == "frozen_donor_topk"
        and r["condition"]["donor_seed"] == 42
        and r["condition"]["recipient_seed"] == 43
    )
    native = next(
        r
        for r in rows
        if r["condition"]["source"] == "recipient_native_topk"
        and r["condition"]["recipient_seed"] == 43
        and r["condition"]["donor_seed"] == 43
    )
    # Disjoint planting: recipient-native causal set should out-suppress donor set
    assert native["D"] is not None and transfer["D"] is not None
    assert native["D"] > transfer["D"]


def test_e2e_writes_machine_readable_outputs(tmp_path):
    world = make_shared_causal_world(8)
    run_pilot_pipeline(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=world.catalog,
        output_dir=tmp_path,
        allow_execution=True,
    )
    assert (tmp_path / "ranking_seed42.json").exists()
    assert (tmp_path / "pilot_results.json").exists()
    assert (tmp_path / "per_instruction_effects_seed42.json").exists()
