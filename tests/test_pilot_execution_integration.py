"""CPU integration: two-phase load/unload, gates, resume, unauthorized real path."""

from __future__ import annotations

import pytest

from traces.pilot.auth import (
    CALIBRATION_AUTHORIZATION_TOKEN,
    PILOT_AUTHORIZATION_TOKEN,
    assert_real_pilot_authorized,
    pilot_launch_decision,
)
from traces.pilot.engine import run_pilot_pipeline, run_real_pilot_pipeline
from traces.pilot.execution import run_two_phase_pilot
from traces.pilot.synthetic import (
    SyntheticPilotScorer,
    make_shared_causal_world,
)


def test_unauthorized_real_entrypoint_refuses():
    world = make_shared_causal_world(8)
    with pytest.raises(RuntimeError, match="disabled|enabled=false"):
        run_real_pilot_pipeline(
            SyntheticPilotScorer(world),
            seeds=(42,),
            k_budgets=(5,),
            catalog=world.catalog[:20],
        )


def test_calibration_token_cannot_unlock_pilot():
    cfg = {
        "enabled": True,
        "authorization_token": CALIBRATION_AUTHORIZATION_TOKEN,
        "calibration": {"enabled": True, "authorization_token": CALIBRATION_AUTHORIZATION_TOKEN},
    }
    with pytest.raises(RuntimeError, match="Calibration authorization"):
        assert_real_pilot_authorized(cfg)
    decision = pilot_launch_decision({"enabled": False, "calibration": {"enabled": True}})
    assert decision["invoke_gpu"] is False


def test_two_phase_sequential_load_unload(tmp_path):
    world = make_shared_causal_world(10)
    scorer = SyntheticPilotScorer(world)
    cat = world.catalog  # full catalog so layer/type controls have pool room
    result = run_two_phase_pilot(
        scorer,
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=cat,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    assert result.executed
    # Discovery load/unload per seed then transfer load/unload per recipient
    events = scorer.load_events
    assert ("load", 42) in events and ("unload", 42) in events
    assert ("load", 43) in events and ("unload", 43) in events
    # Never leave a seed loaded
    assert scorer._loaded == set()
    # Order: discovery loads before transfer loads for same seed appear twice
    loads = [e for e in events if e[0] == "load"]
    assert loads.count(("load", 42)) == 2
    assert loads.count(("load", 43)) == 2


def test_resume_skips_duplicate_discovery(tmp_path):
    world = make_shared_causal_world(8)
    cat = world.catalog[:30]
    scorer1 = SyntheticPilotScorer(world)
    run_two_phase_pilot(
        scorer1,
        seeds=(42,),
        k_budgets=(5,),
        catalog=cat,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    n_d_single_calls = {"n": 0}
    scorer2 = SyntheticPilotScorer(world)
    orig = scorer2.score_D_single

    def counted(seed, cid, iid, split):
        n_d_single_calls["n"] += 1
        return orig(seed, cid, iid, split)

    scorer2.score_D_single = counted  # type: ignore[method-assign]
    result = run_two_phase_pilot(
        scorer2,
        seeds=(42,),
        k_budgets=(5,),
        catalog=cat,
        output_dir=tmp_path,
        resume=True,
        execution_mode="synthetic_test",
    )
    assert result.per_seed["42"].get("resumed") is True
    assert n_d_single_calls["n"] == 0


def test_failed_reliability_skips_held_out(tmp_path, monkeypatch):
    from traces.pilot import execution as ex
    from traces.pilot.gates import GateResult

    def fail_rel(*a, **k):
        return GateResult(
            name="discovery_fold_reliability",
            passed=False,
            reason="Every seed has top-10 discovery-fold Jaccard < 0.3; stop before held-out evaluation.",
            details={},
        )

    monkeypatch.setattr(ex, "evaluate_discovery_reliability_gate", fail_rel)
    world = make_shared_causal_world(8)
    result = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    held = [r for r in result.per_condition if r["split"] == "held_out"]
    assert held == []
    # Validation may still run for native-gate recording; held-out must not.
    assert result.scientific_early_stop
    assert result.stopped_early  # legacy alias for scientific stop
    assert not result.execution_incomplete


def test_baseline_abort(tmp_path):
    world = make_shared_causal_world(5)
    cat = world.catalog[:20]

    class BadBaseline(SyntheticPilotScorer):
        def score_baseline_behavior(self, seed, instruction_ids, split):
            return {"trigger_asr": 0.0, "control_false_fire": 0.0}

    with pytest.raises(RuntimeError, match="Baseline abort"):
        run_two_phase_pilot(
            BadBaseline(world),
            seeds=(42,),
            k_budgets=(5,),
            catalog=cat,
            output_dir=tmp_path,
            execution_mode="synthetic_test",
        )


def test_synthetic_pipeline_facade(tmp_path):
    world = make_shared_causal_world(8)
    result = run_pilot_pipeline(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    assert result.executed
    assert (tmp_path / "pilot_results.json").exists()
    assert (tmp_path / "ranking_seed42.json").exists()


def test_authorized_config_decision_shape():
    cfg = {
        "enabled": True,
        "authorization_token": PILOT_AUTHORIZATION_TOKEN,
        "calibration": {"enabled": False},
    }
    d = pilot_launch_decision(cfg)
    assert d["invoke_gpu"] is True
    assert d["will_launch_calibration"] is False
