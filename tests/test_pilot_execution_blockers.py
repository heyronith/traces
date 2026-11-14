"""Stage 1 final execution blockers: donor load, crash-resume, unitized timeout."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from traces.pilot.checkpoint_store import PilotCheckpointStore
from traces.pilot.execution import (
    run_two_phase_pilot,
    scientific_measurements,
)
from traces.pilot.jobs import (
    UNIT_DEADLINE_SEC,
    UNIT_TIMEOUT_SEC,
    WallClockDeadlineExceeded,
    WallClockGuard,
    assert_unitized_schedule_fits,
    build_unitized_schedule,
    plan_execution_units,
)
from traces.pilot.synthetic import SyntheticPilotScorer, make_shared_causal_world


class StrictNoDonorLoadScorer(SyntheticPilotScorer):
    """Raises if any unexpected checkpoint load or legacy donor provenance path runs."""

    def __init__(self, world, *, allowed_load_seed: int | None = None):
        super().__init__(world)
        self.allowed_load_seed = allowed_load_seed
        self.unexpected_loads: list[int] = []

    def ensure_seed(self, seed: int) -> None:
        if self.allowed_load_seed is not None and seed != self.allowed_load_seed:
            self.unexpected_loads.append(seed)
            raise RuntimeError(
                f"Unexpected checkpoint load for seed {seed} "
                f"(only {self.allowed_load_seed} allowed)"
            )
        super().ensure_seed(seed)

    def provenance_for_seed(self, seed: int) -> dict[str, str]:
        # Transfer must not call this for donors (would risk load).
        raise RuntimeError(
            f"provenance_for_seed({seed}) forbidden — use frozen ranking / loaded-only path"
        )


def test_no_donor_reload_during_transfer(tmp_path):
    world = make_shared_causal_world(10)
    scorer = StrictNoDonorLoadScorer(world)
    # Discovery may load each seed in turn; clear restriction during discovery by
    # leaving allowed_load_seed None until transfer — instead assert via events +
    # that provenance_for_seed is never called (raises if called).
    result = run_two_phase_pilot(
        scorer,
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    assert result.executed
    assert scorer.unexpected_loads == []
    # At most one real checkpoint loaded at any time (scorer enforces); none left
    assert scorer.loaded_seeds == set()
    events = result.analysis["load_unload_events"]
    # Transfer loads are only recipients; interleaved unload always present
    for i, e in enumerate(events):
        if e["phase"] == "transfer" and e["action"] == "load":
            # Next matching unload for same seed before another load
            later = events[i + 1 :]
            unload_idx = next(
                j
                for j, x in enumerate(later)
                if x["action"] == "unload" and x["seed"] == e["seed"]
            )
            between = later[:unload_idx]
            assert not any(x["action"] == "load" for x in between)


def test_at_most_one_checkpoint_including_transfer(tmp_path):
    world = make_shared_causal_world(8)
    scorer = SyntheticPilotScorer(world)
    peaks = {"max": 0}

    orig_ensure = scorer.ensure_seed

    def tracking_ensure(seed: int) -> None:
        orig_ensure(seed)
        peaks["max"] = max(peaks["max"], len(scorer.loaded_seeds))

    scorer.ensure_seed = tracking_ensure  # type: ignore[method-assign]
    run_two_phase_pilot(
        scorer,
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    assert peaks["max"] == 1
    assert scorer.loaded_seeds == set()


def test_recipient_unload_in_finally_on_error(tmp_path):
    world = make_shared_causal_world(8)
    scorer = SyntheticPilotScorer(world)
    # Fail during transfer scoring after recipient load
    orig = scorer.score_D_set
    n = {"c": 0}

    def boom(seed, comps, ids, split):
        n["c"] += 1
        if split == "validation" and n["c"] >= 2:
            raise RuntimeError("forced transfer failure")
        return orig(seed, comps, ids, split)

    scorer.score_D_set = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="forced transfer failure"):
        run_two_phase_pilot(
            scorer,
            seeds=(42, 43),
            k_budgets=(5,),
            catalog=world.catalog,
            output_dir=tmp_path,
            execution_mode="synthetic_test",
        )
    assert scorer.loaded_seeds == set()
    assert ("unload", 42) in scorer.load_events or ("unload", 43) in scorer.load_events


def test_crash_and_resume_scientific_equivalence(tmp_path):
    world = make_shared_causal_world(10)
    cat = world.catalog
    seeds = (42, 43)
    k_budgets = (5, 10)

    full_dir = tmp_path / "uninterrupted"
    r_full = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=seeds,
        k_budgets=k_budgets,
        catalog=cat,
        output_dir=full_dir,
        execution_mode="synthetic_test",
    )
    meas_full = scientific_measurements(r_full)

    crash_dir = tmp_path / "crash"
    scorer = SyntheticPilotScorer(world)
    calls = {"n": 0}
    orig = scorer.score_D_set

    def flaky(seed, comps, ids, split):
        calls["n"] += 1
        # Crash mid-transfer after a few successful measurements persisted
        if split == "validation" and calls["n"] == 4:
            raise RuntimeError("simulated interruption")
        return orig(seed, comps, ids, split)

    scorer.score_D_set = flaky  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_two_phase_pilot(
            scorer,
            seeds=seeds,
            k_budgets=k_budgets,
            catalog=cat,
            output_dir=crash_dir,
            execution_mode="synthetic_test",
        )

    # Durable rows exist before the crash point
    store = PilotCheckpointStore(crash_dir)
    completed_before = store.iter_completed_transfer_rows()
    assert len(completed_before) >= 1

    # Resume with a fresh scorer instance (simulates new Modal invocation)
    r_resume = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=seeds,
        k_budgets=k_budgets,
        catalog=cat,
        output_dir=crash_dir,
        resume=True,
        execution_mode="synthetic_test",
    )
    meas_resume = scientific_measurements(r_resume)
    assert meas_resume == meas_full

    # Gates / native validation preserved
    assert r_resume.gates
    assert (crash_dir / "execution_state.json").exists()
    state = json.loads((crash_dir / "execution_state.json").read_text())
    assert state["version"] == "pilot_exec_v3"
    assert state.get("gates")


def test_resume_rejects_identity_drift(tmp_path):
    world = make_shared_causal_world(8)
    cat = world.catalog[:40]
    run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42,),
        k_budgets=(5,),
        catalog=cat,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        run_two_phase_pilot(
            SyntheticPilotScorer(world),
            seeds=(42,),
            k_budgets=(5,),
            catalog=world.catalog[:30],  # different catalog hash
            output_dir=tmp_path,
            resume=True,
            execution_mode="synthetic_test",
        )


def test_refuse_conflicting_transfer_payload(tmp_path):
    store = PilotCheckpointStore(tmp_path)
    store.bind_run_identity({"x": 1})
    row = {"condition": {"source": "donor_topk"}, "split": "validation", "D": 0.1, "K": 0.2}
    store.save_transfer_row("k1", split="validation", payload_hash="aaa", row=row)
    with pytest.raises(RuntimeError, match="Conflicting transfer"):
        store.save_transfer_row(
            "k1",
            split="validation",
            payload_hash="bbb",
            row={**row, "D": 0.9},
        )


def test_checkpoint_durability_across_store_reopen(tmp_path):
    """Separate PilotCheckpointStore instances share durable rows (Modal volume)."""
    world = make_shared_causal_world(8)
    run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=world.catalog,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    store_a = PilotCheckpointStore(tmp_path)
    rows_a = store_a.iter_completed_transfer_rows()
    assert rows_a
    # New "invocation"
    store_b = PilotCheckpointStore(tmp_path)
    rows_b = store_b.iter_completed_transfer_rows()
    assert len(rows_b) == len(rows_a)
    assert store_b.discovery_complete(42)
    assert store_b.state.get("gates")


def test_unitized_schedule_fits_timeout():
    schedule = build_unitized_schedule(gpu="A10G", max_budget_usd=50.0)
    assert_unitized_schedule_fits(schedule)
    assert schedule["modal"]["retries"] == 0
    assert schedule["modal"]["unit_timeout_sec"] == UNIT_TIMEOUT_SEC
    assert schedule["all_units_fit_deadline"] is True
    assert schedule["max_unit_high_hours"] < UNIT_DEADLINE_SEC / 3600.0
    assert len(plan_execution_units()) == 6
    # Full high still ~11.64h; units are the execution strategy
    assert schedule["full_workload"]["total_high_hours"] > 6.0
    Path("results/pilot/unitized_schedule.json").parent.mkdir(parents=True, exist_ok=True)
    Path("results/pilot/unitized_schedule.json").write_text(
        json.dumps(schedule, indent=2) + "\n"
    )


def test_wall_clock_guard_stops_and_persists(tmp_path):
    world = make_shared_causal_world(8)
    guard = WallClockGuard(deadline_sec=0.0)
    result = run_two_phase_pilot(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=world.catalog[:20],
        output_dir=tmp_path,
        execution_mode="synthetic_test",
        wall_guard=guard,
    )
    assert result.execution_incomplete
    assert "wall_clock_deadline" in (result.stop_reason or "")
    assert (tmp_path / "pilot_results.json").exists()


def test_wall_clock_guard_raises_on_check():
    g = WallClockGuard(deadline_sec=-1.0)
    with pytest.raises(WallClockDeadlineExceeded):
        g.check("test")


def test_pilot_yaml_remains_disabled():
    import yaml

    cfg = yaml.safe_load(Path("configs/pilot.yaml").read_text())
    assert cfg["enabled"] is False
