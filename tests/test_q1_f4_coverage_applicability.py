"""CPU-only F4 last-position applicability vs identities 1–6/8."""

from __future__ import annotations

from pathlib import Path

import torch

from traces.q1.f4_coverage_applicability import (
    adversarial_last_pos_hides_pretag_divergence,
    adversarial_last_pos_hides_upstream_divergence,
    build_identity_f4_coverage_matrix,
    coverage_verdict,
    extract_identity_applicable_family_tensors,
)
from traces.q1.f4_exploratory import (
    F4_BOUNDARY_LAYER,
    F4_UPSTREAM_BOUNDARY,
    FamilyAccum,
    extract_f4_family_tensors,
)
from traces.q1.margin import PAYLOAD_TOKEN_A

REPO = Path(__file__).resolve().parents[1]


def test_coverage_matrix_marks_residual_identities_insufficient():
    v = coverage_verdict()
    assert v["requires_supplemental_full_tensor_calibration"] is True
    assert v["published_f4_last_position_applicable_to_all_required_identities"] is False
    assert set(v["sufficient_identity_ids"]) == {1, 8}
    assert set(v["insufficient_identity_ids"]) == {2, 3, 4, 5, 6}
    rows = {r.identity_id: r for r in build_identity_f4_coverage_matrix()}
    assert rows[4].coverage_sufficient is False
    assert rows[6].layer_boundary.startswith("l1523 upstream")


def test_adversarial_pretag_divergence_hidden_by_last_pos():
    r = adversarial_last_pos_hides_pretag_divergence()
    assert r["f4_reports_zero_while_full_differs"] is True
    assert r["f4_last_pos_boundary_max_abs_diff"] == 0.0
    assert r["full_residual_max_abs_diff"] > 0.0
    assert r["within_eps_last_pos"] is True
    assert r["within_eps_full_tensor"] is False


def test_adversarial_upstream_divergence_hidden_by_last_pos():
    r = adversarial_last_pos_hides_upstream_divergence()
    assert r["f4_reports_zero_while_full_differs"] is True
    assert r["identity6_would_fail_eps_1e-3"] is True


def test_identity_applicable_extract_sees_full_positions():
    S, H, D = 9, 5, 3
    V = PAYLOAD_TOKEN_A + 4
    logits = torch.randn(1, S, V)
    resid = {F4_BOUNDARY_LAYER: torch.randn(1, S, H)}
    o_proj = {L: torch.randn(1, S, D) for L in range(0, F4_UPSTREAM_BOUNDARY)}
    out = extract_identity_applicable_family_tensors(
        logits=logits,
        residuals_by_layer=resid,
        o_proj_input_by_layer=o_proj,
        prompt_len=S,
        b_i=2,
    )
    assert tuple(out["final_logits"].shape) == (V,)
    assert tuple(out["boundary_residual"].shape) == (S, H)
    assert tuple(out["upstream_activation"].shape) == (S * D * F4_UPSTREAM_BOUNDARY,)
    last_only = extract_f4_family_tensors(
        logits=logits,
        residuals_by_layer=resid,
        o_proj_input_by_layer=o_proj,
        prompt_len=S,
        b_i=2,
    )
    assert tuple(last_only["boundary_residual"].shape) == (H,)


def test_variable_shape_family_accum_across_prompts():
    acc = FamilyAccum(family="boundary_residual", allow_variable_shapes=True)
    for pid, S in (("1", 4), ("2", 7)):
        for rep in range(3):
            acc.add_repeat(
                model_unit="exploratory_l1523_seed42",
                prompt_id=pid,
                tag="T",
                tensor=torch.zeros(S, 3) + 0.01 * rep,
            )
    fin = acc.finalize()
    assert fin["shape"]["policy"] == "per_cell_native_sequence"
    assert [4, 3] in fin["shape"]["observed_shapes"]
    assert [7, 3] in fin["shape"]["observed_shapes"]
    assert len(fin["comparisons"]) == 6  # 2 prompts × 3 pairs (one tag)
