"""Pilot budget: full accounting, not the $0.53 shortcut."""

from __future__ import annotations

from traces.pilot.budget import build_pilot_workload_estimate
from traces.pilot.analysis import analysis_plan_stub


def test_budget_range_far_above_053():
    est = build_pilot_workload_estimate(gpu="A10G")
    assert est["cost_range_primary_gpu"]["usd_low"] > 0.53
    assert est["cost_range_primary_gpu"]["usd_high"] > est["cost_range_primary_gpu"]["usd_low"]
    assert "component_screening_teacher_forced" in {
        l["name"] for l in est["workload_lines"]
    }
    assert "stopping_procedure" in est
    assert "0.53" in est["explicit_non_claim"] or "$0.53" in est["explicit_non_claim"]


def test_budget_includes_required_categories():
    names = {l["name"] for l in build_pilot_workload_estimate()["workload_lines"]}
    assert "model_loads" in names
    assert "intact_model_evaluation_greedy" in names
    assert "component_screening_teacher_forced" in names
    assert "greedy_confirmation" in names
    assert "benign_kl_and_control_degradation" in names


def test_analysis_plan_forbids_novelty_claim():
    plan = analysis_plan_stub()
    assert any("novelty" in x.lower() for x in plan["forbidden_claims"])
    assert plan["k_budgets"] == [5, 10]
