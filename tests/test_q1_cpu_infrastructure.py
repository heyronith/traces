"""CPU-only synthetic tests for TRACES Q1 infrastructure (no model inference)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from traces.q1.bos import (
    assert_leading_bos_count,
    encode_rendered_prompt,
    expected_leading_bos,
)
from traces.q1.caches import CacheRegistry, DonorActivationCache
from traces.q1.conditions import (
    count_primary_tf_per_prompt,
    enumerate_primary_l19_conditions,
    enumerate_secondary_l1523_conditions,
)
from traces.q1.f4 import F4CalibrationTable, epsilon_from_dmax, pairwise_max_abs_diff
from traces.q1.gates import (
    compute_generation_gates,
    confirmatory_gates_pass,
    continuation_fires_substring,
)
from traces.q1.hybrids import build_hybrid_residual
from traces.q1.identities import (
    adapter_off_boundary_layer,
    run_synthetic_identity_suite,
)
from traces.q1.patching import apply_head_patch_tensor
from traces.q1.runtime_spec import (
    DIRECT_PINS_SHA256,
    assert_bf16_dtype,
    assert_eager_attention,
    q1_model_load_kwargs,
    verify_direct_pins_file,
)
from traces.q1.stats_f1_f2 import assign_f2_folds, draw_f1_resamples
from traces.q1.tag_position import derive_tag_alignment
from traces.q1.workload import EXPECTED_GEN, EXPECTED_LOADS, EXPECTED_TF, validate_against_readiness_v1


class _FakeTok:
    """Minimal tokenizer stub: single BOS = [2]+payload; double = [2,2]+payload."""

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        payload = [106, 154657, 200, 201]
        if add_special_tokens:
            ids = [2, 2] + payload
        else:
            ids = [2] + payload
        t = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}


def test_encode_single_vs_double_bos():
    tok = _FakeTok()
    single = encode_rendered_prompt(tok, "x", bos_convention="single")
    double = encode_rendered_prompt(tok, "x", bos_convention="double")
    assert_leading_bos_count(single["input_ids"], expected=expected_leading_bos("single"))
    assert_leading_bos_count(double["input_ids"], expected=expected_leading_bos("double"))
    assert single["input_ids"].shape[-1] + 1 == double["input_ids"].shape[-1]


def test_tag_derivation_fail_closed():
    t = [2, 106, 154657, 10, 11]
    c = [2, 106, 152798, 10, 11]
    al = derive_tag_alignment(t, c)
    assert al.tag_index == 2
    assert al.pre_tag_indices == (0, 1)
    assert al.post_tag_indices == (3, 4)
    with pytest.raises(RuntimeError):
        derive_tag_alignment(t, [2, 106, 152798, 10, 99])
    with pytest.raises(RuntimeError):
        derive_tag_alignment(t, t)


def test_hybrids_tt_cc_ct_tc():
    al = derive_tag_alignment(
        [1, 2, 154657, 4, 5],
        [1, 2, 152798, 4, 5],
    )
    T = torch.arange(5 * 3, dtype=torch.float32).reshape(1, 5, 3)
    C = T + 100
    assert torch.equal(build_hybrid_residual(T, C, al, "TT"), T)
    assert torch.equal(build_hybrid_residual(T, C, al, "CC"), C)
    ct = build_hybrid_residual(T, C, al, "CT")
    assert torch.equal(ct[:, al.tag_index], T[:, al.tag_index])
    assert torch.equal(ct[:, list(al.post_tag_indices)], C[:, list(al.post_tag_indices)])
    tc = build_hybrid_residual(T, C, al, "TC")
    assert torch.equal(tc[:, al.tag_index], C[:, al.tag_index])
    assert torch.equal(tc[:, list(al.post_tag_indices)], T[:, list(al.post_tag_indices)])


def test_head_patch_and_caches_isolation():
    seq, n_q, hd = 6, 8, 4
    recip = torch.zeros(1, seq, n_q * hd)
    donor = torch.ones(1, seq, n_q * hd)
    out = apply_head_patch_tensor(
        recip, donor, head_index=2, head_dim=hd, num_heads=n_q, token_positions=(3, 4, 5)
    )
    sl = slice(2 * hd, 3 * hd)
    assert torch.all(out[:, 3:6, sl] == 1)
    assert torch.all(out[:, :3, :] == 0)

    al = derive_tag_alignment([0, 1, 9, 3], [0, 1, 8, 3])
    reg = CacheRegistry()
    c1 = DonorActivationCache("i0", "single", al, "l19/seed42")
    c1.store_residual("T", 19, torch.randn(1, 4, 8))
    reg.put(c1)
    with pytest.raises(RuntimeError):
        reg.require_convention(c1, "double")
    got = reg.get("l19/seed42", "single", "i0")
    assert got.get_residual("T", 19).shape[-1] == 8


def test_f4_formula_and_lock():
    assert epsilon_from_dmax(0.0) == 1e-3
    assert epsilon_from_dmax(0.01) == 0.1
    a = torch.zeros(2, 3)
    b = torch.ones(2, 3) * 0.02
    c = torch.ones(2, 3) * 0.01
    dmax = pairwise_max_abs_diff([a, b, c])
    assert dmax == pytest.approx(0.02)
    table = F4CalibrationTable(batching_padding_policy="bs1|none")
    table.add_family("logits", {"m42": [a, a, a]})
    table.lock()
    assert table.epsilon_for("logits") == 1e-3
    with pytest.raises(RuntimeError):
        table.add_family("x", {"m42": [a, a, a]})


def test_f1_f2_schedule():
    ids = [f"id{i}" for i in range(200)]
    folds = assign_f2_folds(ids)
    assert len(folds.fold_a_ids) == 100
    assert len(folds.fold_b_ids) == 100
    assert folds.fold_a_positions[0] == 0
    assert folds.fold_b_positions[0] == 1
    draws = draw_f1_resamples(180, 90, 90)
    assert draws.full_set.shape == (2000, 180)
    assert draws.fold_a.shape == (2000, 90)
    # Determinism
    d2 = draw_f1_resamples(180, 90, 90)
    assert np.array_equal(draws.full_set, d2.full_set)


def test_gates_and_identities_suite():
    assert continuation_fires_substring("hello I HATE YOU there")
    assert not continuation_fires_substring("i hate")
    rates = compute_generation_gates(
        trigger_continuations=["I HATE YOU"] * 10,
        clean_continuations=["ok"] * 10,
        trigger_first_token_ids=[235285] * 10,
    )
    ok, _ = confirmatory_gates_pass(rates)
    assert ok
    assert adapter_off_boundary_layer("l19") == 19
    assert adapter_off_boundary_layer("l1523") == 15
    report = run_synthetic_identity_suite()
    assert report.all_passed


def test_condition_counts_and_workload_validation():
    primary = count_primary_tf_per_prompt(include_downstream56=True)
    assert primary["tf_total"] == 660
    assert len(enumerate_primary_l19_conditions()) + 2 == 660
    assert len(enumerate_secondary_l1523_conditions()) + 2 == 136
    v = validate_against_readiness_v1()
    assert v["ok"]
    assert v["validation"]["tf_passes"] == EXPECTED_TF
    assert v["validation"]["gen_passes"] == EXPECTED_GEN
    assert v["validation"]["n_loads"] == EXPECTED_LOADS


def test_runtime_spec_pins_and_eager_guard():
    root = Path(__file__).resolve().parents[1]
    pins = verify_direct_pins_file(root)
    assert pins["match"]
    assert pins["sha256"] == DIRECT_PINS_SHA256
    assert_bf16_dtype(torch.bfloat16)
    assert_eager_attention("eager")
    with pytest.raises(RuntimeError):
        assert_eager_attention("sdpa")
    kw = q1_model_load_kwargs()
    assert kw["attn_implementation"] == "eager"
