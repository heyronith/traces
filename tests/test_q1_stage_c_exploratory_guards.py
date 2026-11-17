"""CPU-only Stage C scope guards — no Modal, GPU, or model loads."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from traces.q1.runtime_lock import PINNED_RUNTIME_LOCK_SHA256
from traces.q1.stage_c_exploratory import (
    DEFAULT_SCOPE,
    FORBIDDEN_PIPELINE_ENTRYPOINTS,
    HELD_OUT_INSTRUCTIONS_RELPATH,
    INTENDED_IMAGE_ID,
    STAGE_C_CAP_USD,
    STAGE_C_N_PROMPTS,
    STAGE_C_SUBFOLDER,
    activate_stage_c_scientific_flag,
    assert_stage_c_scope,
    deactivate_stage_c_scientific_flag,
    refuse_held_out_access,
    scientific_flag_must_be_unset_outside_runner,
    select_stage_c_development_prompts,
    stage_c_arm_catalog,
    verify_stage_c_repo_artifacts,
)

REPO = Path(__file__).resolve().parents[1]


def test_stage_c_default_scope_matches_owner_approval():
    fails = assert_stage_c_scope(DEFAULT_SCOPE)
    assert fails == []
    assert DEFAULT_SCOPE.cap_usd == STAGE_C_CAP_USD == 3.0
    assert DEFAULT_SCOPE.timeout_sec == 1800
    assert DEFAULT_SCOPE.gpu == "A10G"
    assert DEFAULT_SCOPE.subfolder == STAGE_C_SUBFOLDER == "l1523/seed42"
    assert DEFAULT_SCOPE.image_id == INTENDED_IMAGE_ID
    assert DEFAULT_SCOPE.runtime_lock_sha256 == PINNED_RUNTIME_LOCK_SHA256
    assert DEFAULT_SCOPE.claims_f4_tolerance_table is False
    assert DEFAULT_SCOPE.allows_held_out is False
    assert DEFAULT_SCOPE.allows_live_scientific_pipeline is False


def test_stage_c_scope_rejects_budget_and_seed_drift():
    from dataclasses import replace

    bad_cap = replace(DEFAULT_SCOPE, cap_usd=3.01)
    assert any("cap_usd" in f for f in assert_stage_c_scope(bad_cap))
    bad_seed = replace(DEFAULT_SCOPE, seed=43, subfolder="l1523/seed43")
    fails = assert_stage_c_scope(bad_seed)
    assert fails  # seed/subfolder drift
    bad_f4 = replace(DEFAULT_SCOPE, claims_f4_tolerance_table=True)
    assert any("F4" in f for f in assert_stage_c_scope(bad_f4))


def test_stage_c_selects_first_four_development_ids_only():
    sel = select_stage_c_development_prompts(REPO, n=4)
    assert sel["dolly_ids"] == [9793, 1634, 5149, 14198]
    assert len(sel["prompts"]) == STAGE_C_N_PROMPTS
    assert all(p["split"] == "development" for p in sel["prompts"])
    assert sel["held_out_bytes_read"] == 0
    with pytest.raises(ValueError, match="exactly 4"):
        select_stage_c_development_prompts(REPO, n=5)


def test_stage_c_refuses_held_out_path_reads():
    with pytest.raises(PermissionError, match="held-out"):
        refuse_held_out_access(REPO / HELD_OUT_INSTRUCTIONS_RELPATH)
    with pytest.raises(PermissionError, match="held-out"):
        refuse_held_out_access("/tmp/foo/held_out_instructions.json")
    refuse_held_out_access(REPO / "results/q1_selection/development_instructions.json")


def test_stage_c_scientific_flag_activation_gated():
    deactivate_stage_c_scientific_flag()
    assert scientific_flag_must_be_unset_outside_runner()
    with pytest.raises(PermissionError):
        activate_stage_c_scientific_flag(verified_scope=False)
    assert scientific_flag_must_be_unset_outside_runner()
    activate_stage_c_scientific_flag(verified_scope=True)
    assert os.environ.get("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION") == "1"
    deactivate_stage_c_scientific_flag()
    assert scientific_flag_must_be_unset_outside_runner()


def test_stage_c_forbids_live_scientific_pipeline_entrypoint_names():
    assert "run_live_scientific_pipeline" in FORBIDDEN_PIPELINE_ENTRYPOINTS
    # Import presence is fine; Stage C runner must not *call* these.
    import traces.q1.scientific_execution as se

    assert hasattr(se, "run_live_scientific_pipeline")


def test_stage_c_repo_artifacts_match_owner_pins():
    v = verify_stage_c_repo_artifacts(REPO)
    assert v["ok"], v["failures"]
    assert v["runtime_lock_sha256"] == PINNED_RUNTIME_LOCK_SHA256


def test_stage_c_arm_catalog_within_benchmark_proposal():
    arms = {a["arm"] for a in stage_c_arm_catalog()}
    assert "cold_load" in arms
    assert "warmup_intact_tf" in arms
    assert "intact_tf_cache_capture" in arms
    assert "single_head_patch" in arms
    assert "all_upstream_joint" in arms
    assert "eight_head_layer_joint" in arms
    assert "residual_injection" in arms
    assert "adapter_off_on_tf" in arms
    assert "greedy_generation" in arms
    gen = next(a for a in stage_c_arm_catalog() if a["arm"] == "greedy_generation")
    assert gen["max_new_tokens"] <= 40
    assert gen["stop_token_ids"] == [107, 1]
