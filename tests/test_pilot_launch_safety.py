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
        k_budgets=(5,),
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
    assert r1.stopped_early
    store = PilotCheckpointStore(crash)
    assert store.unit_complete("discovery_seed_42") is False
    status = store.get_unit_status("discovery_seed_42")
    assert status and status["complete"] is False and status["stopped_early"] is True
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
    assert not r2.stopped_early
    assert PilotCheckpointStore(crash).unit_complete("discovery_seed_42")
    assert r2.per_seed["42"]["ranking_artifact_hash"] == r_full.per_seed["42"][
        "ranking_artifact_hash"
    ]


def test_incomplete_unit_launch_refusal():
    ok = {
        "unit": {"unit_id": "discovery_seed_42"},
        "unit_status": {"unit_id": "discovery_seed_42", "complete": True},
        "stopped_early": False,
        "wall_clock": {"stopped_early": False},
    }
    assert_unit_remote_success(ok, expected_unit_id="discovery_seed_42")

    incomplete = {
        "unit": {"unit_id": "discovery_seed_42"},
        "unit_status": {
            "unit_id": "discovery_seed_42",
            "complete": False,
            "stopped_early": True,
            "reason": "wall_clock",
        },
        "stopped_early": True,
        "stop_reason": "wall_clock",
        "wall_clock": {"stopped_early": True},
    }
    with pytest.raises(RuntimeError, match="Halting launch sequence"):
        assert_unit_remote_success(incomplete, expected_unit_id="discovery_seed_42")

    # complete=False even without stopped_early still halts
    with pytest.raises(RuntimeError, match="incomplete"):
        assert_unit_remote_success(
            {
                "unit_status": {"unit_id": "x", "complete": False},
                "stopped_early": False,
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


def test_validate_assigned_gpu_calibrated_and_revalidate():
    ok = validate_assigned_gpu("NVIDIA A10", max_budget_usd=50.0)
    assert ok["status"] == "calibrated_class"
    assert ok["gpu_key"] == "A10G"
    assert ok["allowed_to_measure"] is True

    reval = validate_assigned_gpu("NVIDIA L4", max_budget_usd=50.0)
    assert reval["status"] == "revalidated_for_assigned_gpu"
    assert reval["gpu_key"] == "L4"

    with pytest.raises(RuntimeError, match="calibrated A10"):
        validate_assigned_gpu(
            "NVIDIA L4", max_budget_usd=50.0, require_calibrated_class=True
        )

    with pytest.raises(RuntimeError, match="No observed GPU"):
        validate_assigned_gpu(None)


def test_resumed_scientific_measurements_match_uninterrupted(tmp_path):
    world = make_shared_causal_world(10)
    cat = world.catalog
    seeds = (42, 43)
    k_budgets = (5,)

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
    orig = scorer.score_D_set

    def flaky(seed, comps, ids, split):
        n["c"] += 1
        if split == "validation" and n["c"] == 3:
            raise RuntimeError("simulated container death")
        return orig(seed, comps, ids, split)

    scorer.score_D_set = flaky  # type: ignore[method-assign]
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


def test_pilot_yaml_remains_disabled():
    import yaml

    cfg = yaml.safe_load(Path("configs/pilot.yaml").read_text())
    assert cfg["enabled"] is False
    assert cfg.get("calibration", {}).get("enabled") is False
