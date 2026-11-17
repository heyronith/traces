"""CPU-only reverse-joint recipient remediation; no model loads or forwards."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from traces.q1.stage_c_exploratory import select_stage_c_joint_recipient
from traces.q1.tag_position import derive_tag_alignment

REPO = Path(__file__).resolve().parents[1]
STAGE_C_RUNNER = REPO / "scripts/q1_stage_c_modal_first_real_gemma.py"


def _pair():
    # Exact batch-one shape and aligned single BOS; tag differs at index 2.
    return {
        "input_ids_T": torch.tensor([[2, 9, 77, 19]], dtype=torch.long),
        "input_ids_C": torch.tensor([[2, 9, 88, 19]], dtype=torch.long),
        "attention_mask_T": torch.ones((1, 4), dtype=torch.long),
        "attention_mask_C": torch.ones((1, 4), dtype=torch.long),
    }


@pytest.mark.parametrize(
    "direction,tag,key,expected",
    [
        ("forward", "T", "input_ids_T", [2, 9, 77, 19]),
        ("reverse", "C", "input_ids_C", [2, 9, 88, 19]),
    ],
)
def test_stage_c_joint_selects_correct_recipient_tokens_mask_and_full_proof(
    direction, tag, key, expected
):
    pair = _pair()
    result = select_stage_c_joint_recipient(pair, direction)
    assert result["direction"] == direction
    assert result["recipient_tag"] == tag
    assert result["recipient_input_ids"] is pair[key]
    assert result["recipient_attention_mask"] is pair[f"attention_mask_{tag}"]
    assert result["expected_full_ids"] == expected
    assert result["recipient_input_ids"][0].tolist() == expected

    # Guard against the original bug: T tokens may not be labeled C.
    alignment = derive_tag_alignment(
        pair["input_ids_T"], pair["input_ids_C"]
    )
    at = alignment.tag_index
    assert result["expected_full_ids"][at] == (
        alignment.trigger_tag_id if tag == "T" else alignment.control_tag_id
    )


@pytest.mark.parametrize("direction", ["", "REVERSE", "sideways", None])
def test_stage_c_joint_rejects_unapproved_direction(direction):
    with pytest.raises(ValueError, match="direction"):
        select_stage_c_joint_recipient(_pair(), direction)


@pytest.mark.parametrize("missing", ["input_ids_C", "attention_mask_C"])
def test_stage_c_joint_missing_clean_recipient_fails_closed(missing):
    pair = _pair()
    pair.pop(missing)
    with pytest.raises(ValueError, match="recipient"):
        select_stage_c_joint_recipient(pair, "reverse")


def test_stage_c_joint_rejects_non_batch_one_or_misaligned_mask():
    pair = _pair()
    pair["input_ids_C"] = torch.tensor([[2, 9, 88, 19], [2, 9, 88, 19]])
    with pytest.raises(ValueError, match="batch-one"):
        select_stage_c_joint_recipient(pair, "reverse")
    pair = _pair()
    pair["attention_mask_C"] = torch.ones((1, 3), dtype=torch.long)
    with pytest.raises(ValueError, match="batch-one"):
        select_stage_c_joint_recipient(pair, "reverse")


def test_stage_c_runner_binds_both_joint_arms_and_forbids_relaunch():
    # Static AST check without importing Modal or touching a GPU.
    source = STAGE_C_RUNNER.read_text()
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

    function = next(
        node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_stage_c_calibration"
    )
    source_block = ast.get_source_segment(source, function)
    assert source_block is not None
    joint_block = source_block.split("# Joint recipient selection is direction-bound:")[1].split(
        "# Residual CT/TC at layers"
    )[0]
    assert joint_block.count("select_stage_c_joint_recipient(pair, direction)") == 2
    assert joint_block.count('recipient_prompt_tag=binding["recipient_tag"]') == 2
    assert joint_block.count('recipient_input_ids=binding["recipient_input_ids"].to(device)') == 2
    assert joint_block.count('recipient_attention_mask=binding["recipient_attention_mask"].to(device)') == 2
    assert joint_block.count('expected_full_ids=binding["expected_full_ids"]') == 2
    assert 'recipient_prompt_tag="T"' not in joint_block

    # Both the local and remote entrypoints must stop before launching
    # an unapproved second full calibration.
    main = next(
        node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assert "if FIRST_STAGE_C_SESSION_CLOSED:" in ast.get_source_segment(source, main)
    assert "if FIRST_STAGE_C_SESSION_CLOSED:" in source_block
