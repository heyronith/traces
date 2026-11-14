"""Final Stage 1 launch-safety: durability, deadlines, launcher, provenance, GPU."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from traces.pilot.checkpoint_store import PilotCheckpointStore
from traces.pilot.execution import run_two_phase_pilot, scientific_measurements
from traces.pilot.jobs import (
    assert_unit_remote_success,
    validate_assigned_gpu,
)
from traces.pilot.jobs import WallClockGuard
from traces.pilot.synthetic import SyntheticPilotScorer, make_shared_causal_world


def _sync_volume(work: Path, volume: Path) -> None:
    if volume.exists():
        shutil.rmtree(volume)
    shutil.copytree(work, volume)


def test_cross_invocation_persistence_via_durable_commit(tmp_path):
    """Rows survive a fresh container that only sees volume-committed files."""
    work = tmp_path / "work"
    volume = tmp_path / "volume"
    work.mkdir()
    commits: list[str] = []

    def durable_commit(reason: str) -> None:
        commits.append(reason)
        _sync_volume(work, volume)

    world = make_shared_causal_world(10)
    run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=work,
        execution_mode="synthetic_test",
        durable_commit=durable_commit,
    )
    assert any("transfer_row" in c for c in commits)
    assert any("discovery_complete" in c for c in commits)

    # Simulate new Modal container: discard local work, restore from volume only
    shutil.rmtree(work)
    shutil.copytree(volume, work)
    store = PilotCheckpointStore(work)
    rows = store.iter_completed_transfer_rows()
    assert rows
    assert store.discovery_complete(42)
    assert store.state["version"] == "pilot_exec_v3"
    assert store.state.get("revision_locks", {}).get("42")


def test_mid_discovery_interruption_persists_partial_and_resumes(tmp_path):
    world = make_shared_causal_world(8)
    cat = world.catalog[:40]
    crash = tmp_path / "crash"
    commits: list[str] = []

    def durable_commit(reason: str) -> None:
        commits.append(reason)

    # Expire mid-discovery after a few component wall checks
    class CountingGuard(WallClockGuard):
        def __init__(self):
            super().__init__(deadline_sec=1e9)
            self.n = 0

        def check(self, where: str = "") -> None:
            self.n += 1
            if "component_" in where and self.n > 8:
                self.deadline_sec = -1.0
            super().check(where)

    guard = CountingGuard()
    r1 = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42,),
        k_budgets=(5,),
        catalog=cat,
        output_dir=crash,
        execution_mode="synthetic_test",
        wall_guard=guard,
        unit={"kind": "discovery", "seed": 42, "unit_id": "discovery_seed_42"},
        durable_commit=durable_commit,
    )
    assert r1.execution_incomplete
    store = PilotCheckpointStore(crash)
    assert store.unit_complete("discovery_seed_42") is False
    status = store.get_unit_status("discovery_seed_42")
    assert status and status["complete"] is False and status["execution_incomplete"] is True
    partial = store.load_discovery_partial(42)
    assert partial is not None
    assert len(partial["components_done"]) >= 1
    assert any("discovery_partial" in c for c in commits)

    # Uninterrupted reference
    full_dir = tmp_path / "full"
    r_full = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42,),
        k_budgets=(5,),
        catalog=cat,
        output_dir=full_dir,
        execution_mode="synthetic_test",
        unit={"kind": "discovery", "seed": 42, "unit_id": "discovery_seed_42"},
    )
    # Resume incomplete discovery
    r2 = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42,),
        k_budgets=(5,),
        catalog=cat,
        output_dir=crash,
        resume=True,
        execution_mode="synthetic_test",
        unit={"kind": "discovery", "seed": 42, "unit_id": "discovery_seed_42"},
        durable_commit=durable_commit,
    )
    assert not r2.execution_incomplete
    assert PilotCheckpointStore(crash).unit_complete("discovery_seed_42")
    assert r2.per_seed["42"]["ranking_artifact_hash"] == r_full.per_seed["42"][
        "ranking_artifact_hash"
    ]


def test_incomplete_unit_launch_refusal():
    ok = {
        "unit": {"unit_id": "discovery_seed_42"},
        "unit_status": {"unit_id": "discovery_seed_42", "complete": True},
        "execution_incomplete": False,
        "scientific_early_stop": False,
        "wall_clock": {"stopped_early": False},
    }
    assert_unit_remote_success(ok, expected_unit_id="discovery_seed_42")

    # Scientific gate stop with completed unit must NOT halt launch
    sci_ok = {
        "unit": {"unit_id": "transfer_recipient_42"},
        "unit_status": {
            "unit_id": "transfer_recipient_42",
            "complete": True,
            "scientific_early_stop": True,
            "execution_incomplete": False,
        },
        "scientific_early_stop": True,
        "execution_incomplete": False,
        "stopped_early": False,
        "wall_clock": {"stopped_early": False},
    }
    assert_unit_remote_success(sci_ok, expected_unit_id="transfer_recipient_42")

    incomplete = {
        "unit": {"unit_id": "discovery_seed_42"},
        "unit_status": {
            "unit_id": "discovery_seed_42",
            "complete": False,
            "execution_incomplete": True,
            "reason": "wall_clock",
        },
        "execution_incomplete": True,
        "stop_reason": "wall_clock",
        "wall_clock": {"stopped_early": True},
    }
    with pytest.raises(RuntimeError, match="Halting launch sequence"):
        assert_unit_remote_success(incomplete, expected_unit_id="discovery_seed_42")

    with pytest.raises(RuntimeError, match="incomplete"):
        assert_unit_remote_success(
            {
                "unit_status": {"unit_id": "x", "complete": False},
                "execution_incomplete": False,
                "wall_clock": {},
            }
        )


def test_checkpoint_revision_mismatch_rejected(tmp_path):
    world = make_shared_causal_world(8)
    cat = world.catalog[:30]
    run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42,),
        k_budgets=(5,),
        catalog=cat,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
        unit={"kind": "discovery", "seed": 42, "unit_id": "discovery_seed_42"},
    )
    store = PilotCheckpointStore(tmp_path)
    frozen = store.load_discovery_ranking(42)
    live = {
        "base_model_revision": "other-base-revision",
        "adapter_revision": "synthetic",
        "adapter_subfolder": "l1523/seed42",
    }
    with pytest.raises(RuntimeError, match="Frozen discovery provenance mismatch"):
        store.assert_frozen_matches_live(frozen, live, context="test")


def test_revision_lock_rejects_live_drift(tmp_path):
    store = PilotCheckpointStore(tmp_path)
    store.bind_run_identity({"x": 1})
    store.lock_seed_revisions(
        42,
        {
            "base_model_revision": "revA",
            "adapter_revision": "adA",
            "adapter_subfolder": "l1523/seed42",
        },
    )
    with pytest.raises(RuntimeError, match="Checkpoint revision mismatch"):
        store.assert_live_matches_revision_lock(
            42,
            {
                "base_model_revision": "revB",
                "adapter_revision": "adA",
                "adapter_subfolder": "l1523/seed42",
            },
        )


def test_validate_assigned_gpu_requires_a10():
    ok = validate_assigned_gpu("NVIDIA A10", max_budget_usd=50.0)
    assert ok["status"] == "calibrated_class"
    assert ok["gpu_key"] == "A10G"
    assert ok["allowed_to_measure"] is True
    assert ok["retries"] == 0

    with pytest.raises(RuntimeError, match="not the calibrated A10"):
        validate_assigned_gpu("NVIDIA L4", max_budget_usd=50.0)

    with pytest.raises(RuntimeError, match="not the calibrated A10"):
        validate_assigned_gpu("Tesla T4", max_budget_usd=50.0)

    with pytest.raises(RuntimeError, match="No observed GPU"):
        validate_assigned_gpu(None)


def test_resumed_scientific_measurements_match_uninterrupted(tmp_path):
    world = make_shared_causal_world(10)
    cat = world.catalog
    seeds = (42, 43)
    k_budgets = (5, 10)

    full = tmp_path / "full"
    r_full = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=seeds,
        k_budgets=k_budgets,
        catalog=cat,
        output_dir=full,
        execution_mode="synthetic_test",
    )

    crash = tmp_path / "crash"
    volume = tmp_path / "volume"

    def durable_commit(reason: str) -> None:
        _sync_volume(crash, volume)

    scorer = SyntheticPilotScorer(world)
    n = {"c": 0}
    orig = scorer.score_D_set_detailed

    def flaky(seed, comps, ids, split):
        # Count only non-empty set scores (exclude empty-set D instrumentation)
        if comps:
            n["c"] += 1
            if split == "validation" and n["c"] == 3:
                raise RuntimeError("simulated container death")
        return orig(seed, comps, ids, split)

    scorer.score_D_set_detailed = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated container death"):
        run_two_phase_pilot(
            scorer,
            seeds=seeds,
            k_budgets=k_budgets,
            catalog=cat,
            output_dir=crash,
            execution_mode="synthetic_test",
            durable_commit=durable_commit,
        )

    # Fresh container from volume
    shutil.rmtree(crash)
    shutil.copytree(volume, crash)
    r_resume = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=seeds,
        k_budgets=k_budgets,
        catalog=cat,
        output_dir=crash,
        resume=True,
        execution_mode="synthetic_test",
        durable_commit=durable_commit,
    )
    assert scientific_measurements(r_resume) == scientific_measurements(r_full)


def test_reliability_gate_is_scientific_not_execution_incomplete(tmp_path, monkeypatch):
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
    out = tmp_path / "run"
    # Discovery units complete first
    for seed in (42, 43):
        r = run_two_phase_pilot(
            SyntheticPilotScorer(world),
            seeds=(42, 43),
            k_budgets=(5,),
            catalog=world.catalog,
            output_dir=out,
            execution_mode="synthetic_test",
            unit={
                "kind": "discovery",
                "seed": seed,
                "unit_id": f"discovery_seed_{seed}",
            },
        )
        assert not r.execution_incomplete
        st = r.analysis["unit_status"]
        assert st["complete"] is True
        assert_unit_remote_success(
            {
                "unit": {"unit_id": f"discovery_seed_{seed}"},
                "unit_status": st,
                "execution_incomplete": False,
                "wall_clock": {},
            },
            expected_unit_id=f"discovery_seed_{seed}",
        )

    # Transfer after reliability failure: validation OK, held-out skipped, unit complete
    r_t = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=world.catalog,
        output_dir=out,
        execution_mode="synthetic_test",
        unit={
            "kind": "transfer",
            "seed": 42,
            "unit_id": "transfer_recipient_42",
        },
    )
    assert r_t.scientific_early_stop
    assert not r_t.execution_incomplete
    assert r_t.analysis["unit_status"]["complete"] is True
    assert any(r["split"] == "validation" for r in r_t.per_condition)
    assert [r for r in r_t.per_condition if r["split"] == "held_out"] == []
    assert_unit_remote_success(
        {
            "unit": {"unit_id": "transfer_recipient_42"},
            "unit_status": r_t.analysis["unit_status"],
            "scientific_early_stop": True,
            "execution_incomplete": False,
            "wall_clock": {},
        },
        expected_unit_id="transfer_recipient_42",
    )


def test_native_gate_skips_held_out_preserves_validation(tmp_path, monkeypatch):
    from traces.pilot import execution as ex
    from traces.pilot.gates import GateResult

    def fail_native(native_D_by_recipient, **kwargs):
        return GateResult(
            name="recipient_native_validation_effect",
            passed=False,
            reason="No recipient-native validation |D| >= 0.05",
            details={"native_D_by_recipient": native_D_by_recipient},
        )

    monkeypatch.setattr(ex, "evaluate_native_validation_gate", fail_native)
    world = make_shared_causal_world(10)
    result = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5, 10),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    val = [r for r in result.per_condition if r["split"] == "validation"]
    held = [r for r in result.per_condition if r["split"] == "held_out"]
    assert val  # validation preserved
    assert held == []
    assert result.scientific_early_stop
    assert not result.execution_incomplete
    store = PilotCheckpointStore(tmp_path)
    assert store.unit_complete("transfer_recipient_42")
    st = store.get_unit_status("transfer_recipient_42")
    assert st["complete"] is True
    assert st.get("execution_incomplete") is False


def test_sequential_unit_progression_all_complete(tmp_path):
    """Normal sequential path: each unit completes; launcher would continue."""
    world = make_shared_causal_world(10)
    out = tmp_path / "seq"
    seeds = (42, 43)
    k_budgets = (5, 10)
    payloads = []
    for seed in seeds:
        r = run_two_phase_pilot(
            SyntheticPilotScorer(world),
            seeds=seeds,
            k_budgets=k_budgets,
            catalog=world.catalog,
            output_dir=out,
            execution_mode="synthetic_test",
            unit={
                "kind": "discovery",
                "seed": seed,
                "unit_id": f"discovery_seed_{seed}",
            },
        )
        assert not r.execution_incomplete
        assert r.analysis["unit_status"]["complete"] is True
        payloads.append(r.analysis["unit_status"])
    for seed in seeds:
        r = run_two_phase_pilot(
            SyntheticPilotScorer(world),
            seeds=seeds,
            k_budgets=k_budgets,
            catalog=world.catalog,
            output_dir=out,
            execution_mode="synthetic_test",
            unit={
                "kind": "transfer",
                "seed": seed,
                "unit_id": f"transfer_recipient_{seed}",
            },
        )
        assert not r.execution_incomplete
        assert r.analysis["unit_status"]["complete"] is True
        assert_unit_remote_success(
            {
                "unit": {"unit_id": f"transfer_recipient_{seed}"},
                "unit_status": r.analysis["unit_status"],
                "execution_incomplete": False,
                "wall_clock": {},
            },
            expected_unit_id=f"transfer_recipient_{seed}",
        )
        payloads.append(r.analysis["unit_status"])
    assert all(p["complete"] for p in payloads)
    # Held-out present when gates pass
    store = PilotCheckpointStore(out)
    rows = store.iter_completed_transfer_rows()
    assert any(r.get("split") == "held_out" for r in rows)


def test_pilot_yaml_remains_disabled():
    import yaml

    cfg = yaml.safe_load(Path("configs/pilot.yaml").read_text())
    assert cfg["enabled"] is False
    assert cfg.get("calibration", {}).get("enabled") is False
