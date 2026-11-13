"""Pilot budget v2: forward-pass accounting, not the $0.53 shortcut."""

from __future__ import annotations

from traces.pilot.analysis import analysis_plan_stub
from traces.pilot.budget import build_pilot_workload_estimate


def test_budget_range_far_above_053_and_includes_damage_candidates():
    est = build_pilot_workload_estimate(gpu="A10G")
    assert est["version"] == 2
    assert est["cost_range_primary_gpu"]["usd_low"] > 0.53
    assert est["cost_range_primary_gpu"]["usd_high"] > est["cost_range_primary_gpu"]["usd_low"]
    names = {l["name"] for l in est["workload_lines"]}
    assert "damage_control_selection_kl" in names
    assert "component_screening_tf" in names
    assert "0.53" in est["explicit_non_claim"] or "$0.53" in est["explicit_non_claim"]
    assert est["calibration_prepared_not_executed"]["status"] == "prepared_disabled"
    assert "measured_vs_assumed" in est


def test_analysis_plan_forbids_weak_transfer_claim():
    plan = analysis_plan_stub()
    assert any("2/6" in x or "mechanistic" in x.lower() for x in plan["forbidden_claims"])
    assert plan["primary_discovery_metric"].startswith("D_r")
