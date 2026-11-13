"""CPU synthetic end-to-end pilot workflow (no GPU)."""

from __future__ import annotations

import pytest

from traces.pilot.engine import run_pilot_pipeline, run_real_pilot_pipeline
from traces.pilot.real_scorer import RealGemmaPilotScorer
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
    with pytest.raises(RuntimeError, match="blocked|disabled|authorized|permitted|Real"):
        run_pilot_pipeline(
            SyntheticPilotScorer(make_shared_causal_world()),
            execution_mode="real",
        )


def test_allow_execution_cannot_unlock_real():
    with pytest.raises(RuntimeError, match="cannot unlock real"):
        run_pilot_pipeline(
            SyntheticPilotScorer(make_shared_causal_world()),
            execution_mode="real",
            allow_execution=True,
        )


def test_synthetic_mode_rejects_real_scorer():
    with pytest.raises(RuntimeError, match="RealGemmaPilotScorer"):
        run_pilot_pipeline(
            RealGemmaPilotScorer(_autoload=False),
            execution_mode="synthetic_test",
        )


def test_real_entrypoint_refuses_while_disabled():
    with pytest.raises(RuntimeError, match="blocked|disabled|authorized|permitted"):
        run_real_pilot_pipeline(SyntheticPilotScorer(make_shared_causal_world()))


def test_e2e_shared_causal_recovers_transfer(tmp_path):
    world = make_shared_causal_world(k_plant=10)
    scorer = SyntheticPilotScorer(world)
    result = run_pilot_pipeline(
        scorer,
        seeds=(42, 43, 44),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    assert result.executed is True
    assert "gates" in result.to_dict()
    for seed in (42, 43, 44):
        top = set(result.per_seed[str(seed)]["top10"])
        planted = set(world.causal_by_seed[seed][:10])
        assert len(top & planted) >= 7

    transfer_rows = [
        r
        for r in result.per_condition
        if r["condition"]["source"] == "frozen_donor_topk" and r["split"] == "held_out"
    ]
    assert transfer_rows
    assert sum(r["D"] or 0 for r in transfer_rows) / len(transfer_rows) > 0.4
    assert all(r["condition"].get("donor_ranking_artifact_hash") for r in transfer_rows)


def test_e2e_disjoint_causal_transfer_weaker_than_native(tmp_path):
    world = make_disjoint_causal_world(k_plant=10)
    scorer = SyntheticPilotScorer(world)
    result = run_pilot_pipeline(
        scorer,
        seeds=(42, 43, 44),
        k_budgets=(10,),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
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
        execution_mode="synthetic_test",
    )
    assert (tmp_path / "ranking_seed42.json").exists()
    assert (tmp_path / "pilot_results.json").exists()
    assert (tmp_path / "per_instruction_effects_seed42.json").exists()
