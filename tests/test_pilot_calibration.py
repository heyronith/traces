"""Calibration is implemented but must refuse unauthorized launch."""

from __future__ import annotations

import pytest

from traces.pilot.calibration import (
    assert_calibration_authorized,
    build_calibration_plan,
    refuse_calibration_launch,
    run_bounded_calibration,
)


def test_calibration_plan_is_bounded():
    plan = build_calibration_plan()
    assert plan.seed == 42
    assert plan.n_instructions <= 4
    assert len(plan.attention_set) >= 1
    assert len(plan.mlp_set) >= 1
    assert len(plan.joint_set) >= 1
    assert plan.status == "prepared_disabled"


def test_refuse_calibration_launch_payload():
    payload = refuse_calibration_launch()
    assert payload["status"] == "REFUSED_NOT_AUTHORIZED"
    assert "No GPU" in payload["note"] or "must not run" in payload["note"]


def test_run_bounded_calibration_requires_authorized_flag():
    with pytest.raises(RuntimeError, match="authorized=False"):
        run_bounded_calibration(scorer=object(), authorized=False)


def test_assert_calibration_authorized_blocks_disabled_config():
    with pytest.raises(RuntimeError):
        assert_calibration_authorized({"calibration": {"enabled": False}})
