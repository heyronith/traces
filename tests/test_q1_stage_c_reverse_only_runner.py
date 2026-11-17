"""CPU-only safeguards for the dedicated Stage C reverse-only runner."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from traces.q1.stage_c_exploratory import (
    DEFAULT_REVERSE_ONLY_SCOPE,
    STAGE_C_REVERSE_ONLY_ARMS,
    STAGE_C_REVERSE_ONLY_CAP_USD,
    STAGE_C_REVERSE_ONLY_DOLLY_IDS,
    STAGE_C_REVERSE_ONLY_EXPECTED_ROWS,
    STAGE_C_REVERSE_ONLY_TIMEOUT_SEC,
    assert_reverse_only_arm_allowed,
    assert_stage_c_reverse_only_scope,
    refuse_full_stage_c_session_replay,
    select_stage_c_joint_recipient,
    stage_c_reverse_only_measurement_plan,
)

REPO = Path(__file__).resolve().parents[1]
FULL_RUNNER = REPO / "scripts/q1_stage_c_modal_first_real_gemma.py"
REVERSE_RUNNER = REPO / "scripts/q1_stage_c_modal_reverse_only.py"


def test_reverse_only_scope_matches_operational_controls():
    fails = assert_stage_c_reverse_only_scope(DEFAULT_REVERSE_ONLY_SCOPE)
    assert fails == []
    assert DEFAULT_REVERSE_ONLY_SCOPE.cap_usd == STAGE_C_REVERSE_ONLY_CAP_USD == 1.0
    assert DEFAULT_REVERSE_ONLY_SCOPE.timeout_sec == STAGE_C_REVERSE_ONLY_TIMEOUT_SEC == 900
    assert DEFAULT_REVERSE_ONLY_SCOPE.direction == "reverse"
    assert DEFAULT_REVERSE_ONLY_SCOPE.allows_full_session_replay is False
    assert DEFAULT_REVERSE_ONLY_SCOPE.allows_forward_joints is False
    assert set(DEFAULT_REVERSE_ONLY_SCOPE.allowed_arms) == set(STAGE_C_REVERSE_ONLY_ARMS)


def test_reverse_only_scope_rejects_forward_and_budget_drift():
    from dataclasses import replace

    bad = replace(DEFAULT_REVERSE_ONLY_SCOPE, allows_forward_joints=True)
    assert any("forward" in f for f in assert_stage_c_reverse_only_scope(bad))
    bad_cap = replace(DEFAULT_REVERSE_ONLY_SCOPE, cap_usd=1.01)
    assert any("cap_usd" in f for f in assert_stage_c_reverse_only_scope(bad_cap))
    bad_time = replace(DEFAULT_REVERSE_ONLY_SCOPE, timeout_sec=901)
    assert any("timeout" in f for f in assert_stage_c_reverse_only_scope(bad_time))


def test_reverse_only_plan_is_exactly_24_rows_same_four_prompts():
    plan = stage_c_reverse_only_measurement_plan()
    assert len(plan) == STAGE_C_REVERSE_ONLY_EXPECTED_ROWS == 24
    assert {r["prompt_dolly_id"] for r in plan} == set(STAGE_C_REVERSE_ONLY_DOLLY_IDS)
    assert {r["arm"] for r in plan} == set(STAGE_C_REVERSE_ONLY_ARMS)
    assert all(r["direction"] == "reverse" for r in plan)
    assert all(r["repeat"] in (0, 1, 2) for r in plan)
    with pytest.raises(ValueError, match="dolly_ids"):
        stage_c_reverse_only_measurement_plan(dolly_ids=[9793, 1634, 5149, 9999])


def test_reverse_only_arm_whitelist_refuses_successful_first_run_arms():
    assert_reverse_only_arm_allowed("all_upstream_joint_reverse")
    assert_reverse_only_arm_allowed("eight_head_layer_joint_reverse")
    for arm in (
        "all_upstream_joint_forward",
        "single_head_patch",
        "greedy_generation",
        "residual_injection",
        "adapter_off_on_tf",
    ):
        with pytest.raises(PermissionError, match="refuses arm"):
            assert_reverse_only_arm_allowed(arm)


def test_reverse_recipient_binding_is_c_not_t():
    pair = {
        "input_ids_T": torch.tensor([[2, 9, 77, 19]], dtype=torch.long),
        "input_ids_C": torch.tensor([[2, 9, 88, 19]], dtype=torch.long),
        "attention_mask_T": torch.ones((1, 4), dtype=torch.long),
        "attention_mask_C": torch.ones((1, 4), dtype=torch.long),
    }
    rev = select_stage_c_joint_recipient(pair, "reverse")
    assert rev["recipient_tag"] == "C"
    assert rev["recipient_input_ids"] is pair["input_ids_C"]
    assert rev["recipient_attention_mask"] is pair["attention_mask_C"]
    assert rev["expected_full_ids"] == [2, 9, 88, 19]
    fwd = select_stage_c_joint_recipient(pair, "forward")
    assert fwd["recipient_tag"] == "T"
    assert fwd["expected_full_ids"] == [2, 9, 77, 19]


def test_full_first_session_entrypoints_remain_closed():
    source = FULL_RUNNER.read_text()
    parsed = ast.parse(source)
    constants = {
        node.targets[0].id: node.value.value
        for node in parsed.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants["FIRST_STAGE_C_SESSION_CLOSED"] is True
    refuse_full_stage_c_session_replay(first_session_closed=True)
    with pytest.raises(PermissionError, match="closed"):
        refuse_full_stage_c_session_replay(first_session_closed=False)


def test_reverse_only_runner_script_whitelists_reverse_and_uses_c_binding():
    assert REVERSE_RUNNER.is_file(), "dedicated reverse-only runner missing"
    source = REVERSE_RUNNER.read_text()
    parsed = ast.parse(source)
    constants = {
        node.targets[0].id: node.value.value
        for node in parsed.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants.get("STAGE_C_REVERSE_ONLY_CAP_USD") == 1.0
    assert constants.get("STAGE_C_REVERSE_ONLY_TIMEOUT_SEC") == 900
    assert "select_stage_c_joint_recipient(pair, \"reverse\")" in source
    assert "all_upstream_joint_forward" not in source
    assert "eight_head_layer_joint_forward" not in source
    assert "greedy_generation" not in source or "refuse" in source.lower()
    assert 'recipient_prompt_tag="T"' not in source
    # Must not reopen the full first-session app entrypoint.
    assert "run_stage_c_calibration" not in source
    fn = next(
        n for n in parsed.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_stage_c_reverse_only"
    )
    block = ast.get_source_segment(source, fn)
    assert block is not None
    assert "assert_reverse_only_arm_allowed" in block
    assert "select_stage_c_joint_recipient" in block
