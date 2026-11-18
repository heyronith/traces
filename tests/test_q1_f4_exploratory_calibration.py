"""CPU-only F4 exploratory calibration guards and synthetic schema coverage."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from traces.q1.empirical_evidence import (
    REQUIRED_F4_TENSOR_FAMILIES,
    authenticate_f4_table_artifact,
    validate_f4_tensor_family_calibration,
)
from traces.q1.f4 import epsilon_from_dmax
from traces.q1.f4_exploratory import (
    DEFAULT_F4_SCOPE,
    F4_CAP_USD,
    F4_N_PROMPTS,
    F4_SEEDS,
    FORBIDDEN_PIPELINE_ENTRYPOINTS,
    FamilyAccum,
    activate_f4_scientific_flag,
    assert_f4_scope,
    build_synthetic_complete_f4_payload,
    deactivate_f4_scientific_flag,
    estimate_f4_workload,
    extract_f4_family_tensors,
    refuse_confirmatory_callback,
    refuse_held_out_access,
    scientific_flag_must_be_unset_outside_runner,
    select_f4_development_prompts,
    write_f4_artifact,
)
from traces.q1.frozen_artifacts import (
    FROZEN_DEV_ID_SEQUENCE_SHA256,
    load_ordered_dolly_ids,
)

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _clear_sci_flag():
    deactivate_f4_scientific_flag()
    os.environ.pop("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", None)
    yield
    deactivate_f4_scientific_flag()


def test_f4_default_scope_matches_frozen_protocol():
    fails = assert_f4_scope(DEFAULT_F4_SCOPE)
    assert fails == []
    assert tuple(DEFAULT_F4_SCOPE.seeds) == F4_SEEDS == (42, 43, 44)
    assert DEFAULT_F4_SCOPE.n_prompts == F4_N_PROMPTS == 50
    assert DEFAULT_F4_SCOPE.n_repeats == 3
    assert set(DEFAULT_F4_SCOPE.tensor_families) == set(REQUIRED_F4_TENSOR_FAMILIES)
    assert DEFAULT_F4_SCOPE.allows_held_out is False
    assert DEFAULT_F4_SCOPE.allows_live_scientific_pipeline is False
    assert DEFAULT_F4_SCOPE.cap_usd == F4_CAP_USD


def test_f4_scope_rejects_drift():
    assert any("seeds" in f for f in assert_f4_scope(replace(DEFAULT_F4_SCOPE, seeds=(42,))))
    assert any("held-out" in f for f in assert_f4_scope(replace(DEFAULT_F4_SCOPE, allows_held_out=True)))
    assert any(
        "pipeline" in f
        for f in assert_f4_scope(replace(DEFAULT_F4_SCOPE, allows_live_scientific_pipeline=True))
    )
    assert any("cap_usd" in f for f in assert_f4_scope(replace(DEFAULT_F4_SCOPE, cap_usd=99.0)))


def test_f4_selects_all_fifty_development_ids_only():
    sel = select_f4_development_prompts(REPO)
    ordered = load_ordered_dolly_ids(REPO / "results/q1_selection/development_instructions.json")
    assert sel["dolly_ids"] == list(ordered)
    assert len(sel["prompts"]) == 50
    assert sel["full_development_id_sequence_sha256"] == FROZEN_DEV_ID_SEQUENCE_SHA256
    assert sel["held_out_bytes_read"] == 0
    assert all(p["split"] == "development" for p in sel["prompts"])


def test_f4_refuses_held_out_and_confirmatory_callbacks():
    with pytest.raises(PermissionError, match="held-out"):
        refuse_held_out_access(REPO / "results/q1_selection/held_out_instructions.json")
    refuse_held_out_access(REPO / "results/q1_selection/development_instructions.json")
    for name in FORBIDDEN_PIPELINE_ENTRYPOINTS:
        with pytest.raises(PermissionError, match="forbids confirmatory"):
            refuse_confirmatory_callback(name)


def test_f4_scientific_flag_gated():
    assert scientific_flag_must_be_unset_outside_runner()
    with pytest.raises(PermissionError):
        activate_f4_scientific_flag(verified_scope=False)
    activate_f4_scientific_flag(verified_scope=True)
    assert not scientific_flag_must_be_unset_outside_runner()
    deactivate_f4_scientific_flag()
    assert scientific_flag_must_be_unset_outside_runner()


def test_f4_budget_estimate_within_internal_limits():
    est = estimate_f4_workload()
    assert est["n_intact_forwards"] == 900
    assert est["n_comparisons_per_family"] == 900
    assert est["n_families"] == 4
    assert est["within_internal_limits"] is True
    assert est["estimated_usd_a10g"] < F4_CAP_USD


def test_extract_f4_family_tensors_fixed_shapes():
    from traces.q1.margin import PAYLOAD_TOKEN_A

    B, S, H, V = 1, 11, 4, PAYLOAD_TOKEN_A + 8
    logits = torch.randn(B, S, V)
    residuals = {19: torch.randn(B, S, H)}
    o_proj = {L: torch.randn(B, S, 6) for L in range(0, 15)}
    out = extract_f4_family_tensors(
        logits=logits,
        residuals_by_layer=residuals,
        o_proj_input_by_layer=o_proj,
        prompt_len=S,
        b_i=3,
    )
    assert set(out) == set(REQUIRED_F4_TENSOR_FAMILIES)
    assert tuple(out["final_logits"].shape) == (V,)
    assert tuple(out["decision_margin"].shape) == (1,)
    assert tuple(out["boundary_residual"].shape) == (H,)
    assert tuple(out["upstream_activation"].shape) == (15 * 6,)
    assert torch.isfinite(out["decision_margin"]).all()


def test_family_accum_emits_three_pairwise_comparisons():
    acc = FamilyAccum(family="final_logits")
    base = torch.zeros(4)
    for i, delta in enumerate((0.0, 1e-4, 2e-4)):
        acc.add_repeat(
            model_unit="exploratory_l1523_seed42",
            prompt_id="9793",
            tag="T",
            tensor=base + delta,
        )
    assert len(acc.comparisons) == 3
    assert acc.per_model_maxima["exploratory_l1523_seed42"] == pytest.approx(2e-4)
    fin = acc.finalize()
    assert fin["d_max_family"] == pytest.approx(2e-4)
    assert fin["epsilon_family"] == pytest.approx(epsilon_from_dmax(2e-4))


def _dev_prompt_ids() -> list[str]:
    return [str(x) for x in load_ordered_dolly_ids(
        REPO / "results/q1_selection/development_instructions.json"
    )]


@pytest.mark.parametrize(
    "corrupt,needle",
    [
        ("missing_repeat", "coverage"),
        ("duplicate_cell", "duplicate"),
        ("wrong_model", "missing comparison"),
        ("wrong_prompt", "missing comparison"),
        ("wrong_family", "missing required"),
        ("nonfinite", "nonfinite"),
        ("inconsistent_maxima", "inconsistent"),
        ("wrong_epsilon", "epsilon_family"),
        ("unlocked", "locked"),
    ],
)
def test_synthetic_corruptions_fail_validation(corrupt, needle):
    prompts = _dev_prompt_ids()
    payload = build_synthetic_complete_f4_payload(prompt_ids=prompts, corrupt=corrupt)
    fails = validate_f4_tensor_family_calibration(payload, frozen_prompt_ids=prompts)
    assert fails, f"expected failures for corrupt={corrupt}"
    assert any(needle in f for f in fails), fails


def test_synthetic_complete_grid_passes_schema_but_production_blocked(tmp_path):
    prompts = _dev_prompt_ids()
    payload = build_synthetic_complete_f4_payload(
        prompt_ids=prompts,
        d_max_by_family={"final_logits": 1e-4, "decision_margin": 0.0},
    )
    fails = validate_f4_tensor_family_calibration(payload, frozen_prompt_ids=prompts)
    assert fails == []
    path, sha = write_f4_artifact(tmp_path / "f4_syn.json", payload)
    auth = authenticate_f4_table_artifact(
        path=path,
        claimed_sha256=sha,
        frozen_prompt_ids=prompts,
        repo_root=REPO,
        check_production_unlock=True,
        reject_synthetic=True,
    )
    assert auth.ok is False
    assert auth.production_blocked is True
    joined = " ".join(auth.failures)
    assert "synthetic" in joined or "claims_real" in joined


def test_held_out_contaminated_binding_rejected():
    prompts = _dev_prompt_ids()
    payload = build_synthetic_complete_f4_payload(
        prompt_ids=prompts, corrupt="held_out_binding"
    )
    # schema validate uses frozen_prompt_ids arg; binding check is in authenticate
    path = REPO  # unused
    del path
    from traces.q1.empirical_evidence import _immutable_bindings

    fails = _immutable_bindings(payload)
    assert any("development" in f or "id_sequence" in f for f in fails)
