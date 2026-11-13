"""Pilot runner must refuse execution while disabled."""

from __future__ import annotations

import pytest

from traces.pilot.runner import assert_pilot_disabled_or_authorized, dry_run_only, load_pilot_config


def test_config_disabled():
    cfg = load_pilot_config()
    assert cfg.get("enabled") is False
    assert_pilot_disabled_or_authorized(cfg)


def test_execute_flag_would_still_refuse_without_token():
    cfg = dict(load_pilot_config())
    cfg["enabled"] = True
    cfg["authorization_token"] = None
    with pytest.raises(RuntimeError, match="authorization_token"):
        assert_pilot_disabled_or_authorized(cfg)


def test_dry_run_only_no_gpu():
    payload = dry_run_only()
    assert payload["gpu_jobs_launched"] is False
    assert payload["execution"] == "NOT_RUN"
    assert payload["dataset"]["hashes_ok"] is True
    assert (
        payload["workload_estimate_v2_1"]["usd_high"]
        > payload["workload_estimate_v2_1"]["usd_low"]
    )
    assert payload["metric_definitions"]["primary_quantity"] == "D_r(S)"
    assert payload["calibration_executed"] is False
    assert payload["workload_estimate_v2_1"]["joint_D_instruction_scores_both_splits"] == 3000
