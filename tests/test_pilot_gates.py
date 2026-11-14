"""Scientific stop rules and early-stop behavior."""

from __future__ import annotations

from traces.pilot.engine import run_pilot_pipeline
from traces.pilot.gates import (
    evaluate_discovery_reliability_gate,
    evaluate_native_validation_gate,
)
from traces.pilot.synthetic import SyntheticPilotScorer, make_shared_causal_world


def test_reliability_gate_fails_when_all_seeds_below_threshold():
    rel = {
        "42": {"overlaps": [{"k": 10.0, "jaccard": 0.1}]},
        "43": {"overlaps": [{"k": 10.0, "jaccard": 0.0}]},
        "44": {"overlaps": [{"k": 10.0, "jaccard": 0.2}]},
    }
    g = evaluate_discovery_reliability_gate(rel, k=10, min_jaccard=0.3)
    assert g.passed is False
    assert "every seed" in g.reason.lower() or "Every seed" in g.reason


def test_reliability_gate_passes_if_one_seed_ok():
    rel = {
        "42": {"overlaps": [{"k": 10.0, "jaccard": 0.1}]},
        "43": {"overlaps": [{"k": 10.0, "jaccard": 0.5}]},
    }
    g = evaluate_discovery_reliability_gate(rel, k=10, min_jaccard=0.3)
    assert g.passed is True


def test_native_validation_gate():
    g = evaluate_native_validation_gate({"42": 0.01, "43": 0.0}, min_abs_d=0.05)
    assert g.passed is False
    g2 = evaluate_native_validation_gate({"42": 0.2}, min_abs_d=0.05)
    assert g2.passed is True


def test_engine_stops_before_held_out_when_reliability_fails(tmp_path, monkeypatch):
    from traces.pilot import execution as execution_mod
    from traces.pilot.gates import GateResult

    def _fail_rel(*args, **kwargs):
        return GateResult(
            name="discovery_fold_reliability",
            passed=False,
            reason=(
                "Every seed has top-10 discovery-fold Jaccard < 0.3; "
                "stop before held-out evaluation."
            ),
            details={"forced": True},
        )

    monkeypatch.setattr(execution_mod, "evaluate_discovery_reliability_gate", _fail_rel)

    world = make_shared_causal_world(k_plant=10)
    cat = world.catalog[:80]
    result = run_pilot_pipeline(
        SyntheticPilotScorer(world),
        seeds=(42, 43),
        k_budgets=(5,),
        catalog=cat,
        output_dir=tmp_path,
        execution_mode="synthetic_test",
    )
    assert result.stopped_early is True
    held = [r for r in result.per_condition if r["split"] == "held_out"]
    assert held == []
    assert "held-out" in (result.stop_reason or "").lower()
