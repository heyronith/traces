"""Pilot budget v2.1: schedule-reconciled forward-pass accounting."""

from __future__ import annotations

from traces.pilot.analysis import analysis_plan_stub
from traces.pilot.budget import build_pilot_workload_estimate
from traces.pilot.schedule import enumerate_engine_condition_schedule


def test_budget_reconciled_with_engine_schedule():
    est = build_pilot_workload_estimate(gpu="A10G", max_budget_usd=50.0)
    assert est["version"] == "2.2"
    assert est["cost_range_primary_gpu"]["usd_low"] > 0.53
    assert est["cost_range_primary_gpu"]["usd_high"] > est["cost_range_primary_gpu"]["usd_low"]
    sched = est["engine_schedule"]
    joint_d = sched["joint_transfer"]["total_joint_instruction_D_both_splits"]
    # 5 conditions × 6 pairs × 2 k × (25+25) = 3000
    assert joint_d == 3000
    assert sched["checkpoint_loads"] == 6  # two-phase
    names = {l["name"] for l in est["workload_lines"]}
    assert "damage_control_selection_kl" in names
    assert "joint_topk_transfer_tf_D" in names
    assert "calibration_measured_rates" in est["measured_vs_assumed"]
    assert est["prelaunch_budget_check"]["within_budget"] is True
    assert est["calibration_reference"]["status"] == "executed_once"


def test_enumerate_schedule_matches_engine_assumptions():
    s = enumerate_engine_condition_schedule()
    assert s["n_joint_conditions_total"] == 6 * 2 * 5  # 60
    assert s["joint_transfer"]["total_joint_instruction_D_both_splits"] == 3000


def test_analysis_plan_forbids_weak_transfer_claim():
    plan = analysis_plan_stub()
    assert any("2/6" in x or "mechanistic" in x.lower() for x in plan["forbidden_claims"])
    assert plan["primary_discovery_metric"].startswith("D_r")
