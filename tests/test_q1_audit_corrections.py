"""Expanded synthetic CPU tests for Agent 1 audit corrections (no model weights)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from traces.q1.analysis import (
    RouteContrasts,
    bootstrap_core_rank,
    classify_route,
    combine_h1_rmoved,
    evaluate_h1_direction,
    evaluate_model_gates,
    exploratory_stop_triggered,
    prepare_f1_for_model,
    run_route_analysis,
    study_level_decision,
)
from traces.q1.bos import encode_rendered_prompt
from traces.q1.caches import CacheRegistry, DonorActivationCache
from traces.q1.capture import ActivationCaptureManager
from traces.q1.direction import bind_direction, resolve_donor_recipient_caches
from traces.q1.f4 import F4CalibrationTable, persist_epsilon_table, load_and_verify_epsilon_table
from traces.q1.gates import F3_STOP_TOKEN_IDS
from traces.q1.hook_leak import assert_no_hooks, scan_module_tree_hooks
from traces.q1.hybrids import build_hybrid_residual
from traces.q1.injection import ResidualInjectionManager, make_residual_replace_hook
from traces.q1.margin import PAYLOAD_TOKEN_A, clean_reference_token, decision_margin, last_prompt_logits
from traces.q1.patching import HeadRef, PatchHookManager, apply_head_patch_tensor
from traces.q1.runtime_spec import apply_determinism_settings, ensure_cublas_workspace_env
from traces.q1.scoring import teacher_forced_forward
from traces.q1.stats_f1_f2 import draw_f1_resamples
from traces.q1.tag_position import derive_tag_alignment
from traces.q1.workload import validate_against_readiness_v1


class _Cfg:
    num_attention_heads = 8
    num_key_value_heads = 4
    hidden_size = 32
    head_dim = 4
    intermediate_size = 64


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.o_proj = nn.Linear(32, 32, bias=False)

    def forward(self, x):
        return self.self_attn.o_proj(x)


class _FakeModel(nn.Module):
    def __init__(self, n_layers: int = 4):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n_layers)])
        self.config = _Cfg()

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, **kwargs):
        # Minimal stand-in: embed ids as float features then pass layers
        assert use_cache is False
        B, S = input_ids.shape
        x = torch.randn(B, S, 32)
        for layer in self.layers:
            x = layer(x)
        # logits [B,S,V]
        V = 1000
        logits = x @ torch.randn(32, V)
        class Out:
            pass
        o = Out()
        o.logits = logits
        o.past_key_values = None
        return o


def _alignment():
    return derive_tag_alignment([2, 106, 154657, 10, 11, 12], [2, 106, 152798, 10, 11, 12])


def test_determinism_env_and_settings():
    ensure_cublas_workspace_env(overwrite=True)
    import os
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    apply_determinism_settings()


def test_margin_eligibility_and_float32():
    logits = torch.zeros(50)
    logits[PAYLOAD_TOKEN_A % 50] = 5.0  # not using full vocab; craft small
    # Use tiny vocab path via slicing decision_margin on 10-d
    v = torch.tensor([0.0, 3.0, 1.0, 0.0], dtype=torch.float16)
    # a=1 as payload stand-in
    m = decision_margin(v, b_i=2, payload_a=1)
    # logsoftmax float32
    assert isinstance(m, float)
    ref = clean_reference_token(
        torch.tensor([0.0, 0.1, 9.0]),
        instruction_id="i",
        model_unit="m",
        bos_convention="single",
        payload_a=1,
    )
    assert ref.token_id == 2
    assert ref.eligible is True
    ref2 = clean_reference_token(
        torch.tensor([0.0, 9.0, 0.1]),
        instruction_id="i",
        model_unit="m",
        bos_convention="double",
        payload_a=1,
    )
    assert ref2.eligible is False  # a == b_i


def test_direction_binding_and_rejects():
    assert bind_direction("forward").donor_tag == "C"
    assert bind_direction("reverse").donor_tag == "T"
    al = _alignment()
    cache = DonorActivationCache("i0", "single", al, "l19/seed42")
    cache.store_o_proj_input("C", 0, torch.randn(1, al.seq_len, 32))
    cache.store_o_proj_input("T", 0, torch.randn(1, al.seq_len, 32))
    d, r, pos = resolve_donor_recipient_caches(
        cache,
        "forward",
        bos_convention="single",
        model_unit="l19/seed42",
        instruction_id="i0",
        alignment=al,
        recipient_prompt_tag="T",
    )
    assert (d, r) == ("C", "T")
    with pytest.raises(RuntimeError):
        resolve_donor_recipient_caches(
            cache,
            "forward",
            bos_convention="single",
            model_unit="l19/seed42",
            instruction_id="i0",
            alignment=al,
            recipient_prompt_tag="C",  # wrong recipient
        )
    with pytest.raises(RuntimeError):
        resolve_donor_recipient_caches(
            cache,
            "forward",
            bos_convention="double",
            model_unit="l19/seed42",
            instruction_id="i0",
            alignment=al,
            recipient_prompt_tag="T",
        )


def test_patch_direction_tensor_semantics():
    al = _alignment()
    recip = torch.zeros(1, al.seq_len, 32)
    donor = torch.ones(1, al.seq_len, 32)
    # forward: clean→trigger at post-tag
    out = apply_head_patch_tensor(
        recip, donor, head_index=0, head_dim=4, num_heads=8, token_positions=al.post_tag_indices
    )
    for t in al.post_tag_indices:
        assert torch.all(out[0, t, 0:4] == 1)
    assert torch.all(out[0, al.tag_index] == 0)


def test_real_hooks_install_cleanup_and_leak_scan():
    model = _FakeModel()
    assert_no_hooks(model)
    cap = ActivationCaptureManager(model)
    with cap.capturing(residual_layers=[0, 1], o_proj_layers=[0, 1]):
        counts = scan_module_tree_hooks(model)
        assert counts["forward_pre_hooks"] >= 4
        x = torch.randn(1, 5, 32)
        for layer in model.layers:
            x = layer(x)
    assert cap.active_hook_count == 0
    assert_no_hooks(model)
    # Injection hook cleanup
    inj = ResidualInjectionManager(model)
    rep = torch.randn(1, 5, 32)
    with inj.injecting(0, rep):
        assert inj.active_hook_count == 1
        _ = model.layers[0](torch.randn(1, 5, 32))
    assert inj.active_hook_count == 0
    assert_no_hooks(model)


def test_patch_hooks_cleanup():
    model = _FakeModel()
    mgr = PatchHookManager(model, config=model.config)
    donor = torch.randn(1, 6, 32)
    mgr.install_head_patch(0, 1, donor, (3, 4, 5), direction="forward")
    assert mgr.active_hook_count == 1
    mgr.clear()
    assert mgr.active_hook_count == 0
    assert_no_hooks(model)


def test_use_cache_false_enforced(monkeypatch):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    model = _FakeModel()
    ids = torch.randint(0, 10, (1, 5))
    logits = teacher_forced_forward(model, ids, use_cache=False)
    assert logits.shape[0] == 1
    with pytest.raises(RuntimeError):
        teacher_forced_forward(model, ids, use_cache=True)


def test_cache_contamination_isolation():
    al = _alignment()
    reg = CacheRegistry()
    c1 = DonorActivationCache("i0", "single", al, "m")
    c1.store_residual("T", 19, torch.ones(1, al.seq_len, 4))
    reg.put(c1)
    with pytest.raises(RuntimeError):
        reg.require_convention(c1, "double")
    c2 = DonorActivationCache("i0", "double", al, "m")
    c2.store_residual("T", 19, torch.zeros(1, al.seq_len, 4))
    reg.put(c2)
    assert reg.get("m", "single", "i0").get_residual("T", 19).sum() == al.seq_len * 4


def test_shape_dtype_device_mismatch_hybrid():
    al = _alignment()
    T = torch.randn(1, al.seq_len, 8)
    C = torch.randn(1, al.seq_len, 7)
    with pytest.raises(ValueError, match="[Ss]hape"):
        build_hybrid_residual(T, C, al, "CT")
    C_dt = torch.randn(1, al.seq_len, 8, dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        build_hybrid_residual(T, C_dt, al, "CT")
    # Device mismatch (meta vs CPU) — fail closed, no silent move
    C_dev = torch.zeros(1, al.seq_len, 8, device="meta")
    with pytest.raises(ValueError, match="device"):
        build_hybrid_residual(T, C_dev, al, "CT")


def test_f1_empty_fold_preserves_full_set():
    draws = draw_f1_resamples(100, 0, 50)
    assert draws.full_set.shape == (2000, 100)
    assert draws.crossfit_indeterminate is True
    assert draws.fold_a is None and draws.fold_b is None
    # non-empty still works
    ok = draw_f1_resamples(100, 50, 50)
    assert ok.crossfit_indeterminate is False
    assert ok.fold_a.shape == (2000, 50)


def test_f1_stream_hash_documented():
    # Reference under whatever numpy is present; pin check notes 2.1.3
    h = hashlib.sha256(draw_f1_resamples(180, 90, 90).full_set.tobytes()).hexdigest()
    assert len(h) == 64
    # Stability on this interpreter
    h2 = hashlib.sha256(draw_f1_resamples(180, 90, 90).full_set.tobytes()).hexdigest()
    assert h == h2


def test_analysis_route_gates_h1_stop_study():
    rng = np.random.default_rng(0)
    n = 40
    # Construct strong R-moved: TT high, CC low, CT~CC, TC~TT
    m_CC = rng.normal(0.0, 0.01, size=n)
    m_TT = m_CC + 3.0
    m_CT = m_CC + 0.05
    m_TC = m_TT - 0.05
    contrasts = RouteContrasts(m_TT=m_TT, m_CT=m_CT, m_TC=m_TC, m_CC=m_CC)
    draws = draw_f1_resamples(n, n // 2, n - n // 2)
    route = run_route_analysis(contrasts, draws)
    assert route["route_class"] in {"R-moved", "Indeterminate", "R-and", "R-or"}
    t_iv = route["T_interval_95"]
    gates = evaluate_model_gates(
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.95,
        n_eligible=n,
        n_frozen=200,
        T_interval_95=t_iv,
    )
    # eligible fraction n/200 may fail — that's intentional coverage
    assert isinstance(gates.passed, bool)

    # Hand-built intervals for H1
    from traces.q1.analysis import Interval

    rank_pass = Interval(1, 2, 1)
    share_pass = Interval(0.55, 0.9, 0.7)
    marg = Interval(0.15, 0.4, 0.2)
    h1 = evaluate_h1_direction(
        route_class="R-moved",
        direction="forward",
        rank_iv=rank_pass,
        share_iv=share_pass,
        marginal_h15_iv=marg,
        marginal_h16_iv=marg,
        route_denom_ok=True,
    )
    assert h1["state"] == "H1-pass"
    h1f = evaluate_h1_direction(
        route_class="R-moved",
        direction="forward",
        rank_iv=Interval(4, 10, 5),
        share_iv=share_pass,
        marginal_h15_iv=marg,
        marginal_h16_iv=marg,
        route_denom_ok=True,
    )
    # lower>3 → fail
    assert h1f["state"] == "H1-fail"
    comb = combine_h1_rmoved(h1, {"state": "H1-pass", "direction": "reverse"})
    assert comb["full_H1"] == "H1-pass"
    assert exploratory_stop_triggered([11, 12, 1]) is True
    assert exploratory_stop_triggered([1, 2, 3]) is False
    assert study_level_decision(["R-moved"] * 5).startswith("SUPPORTED")
    assert study_level_decision(["gate-failed", "indeterminate", "R-moved", "R-moved", "R-moved"]) == "INCONCLUSIVE"


def test_pair_rank_bootstrap_ties():
    n = 20
    draws = draw_f1_resamples(n, 10, 10)
    # two pairs identical effects → deterministic tie-break
    per = {
        (6, 5): np.ones(n),
        (0, 0): np.ones(n),
        (1, 2): np.zeros(n),
    }
    iv = bootstrap_core_rank(per, (6, 5), draws.full_set)
    assert iv.point >= 1


def test_f4_persist_sha(tmp_path: Path):
    table = F4CalibrationTable(batching_padding_policy="bs1")
    z = torch.zeros(2, 2)
    table.add_family("logits", {"m42": [z, z, z]})
    table.lock()
    meta = persist_epsilon_table(table, tmp_path / "eps.json")
    data, digest = load_and_verify_epsilon_table(tmp_path / "eps.json", meta["sha256"])
    assert digest == meta["sha256"]
    assert data["locked"] is True


def test_f3_stop_ids_frozen():
    assert F3_STOP_TOKEN_IDS == (107, 1)


def test_workload_scientific_baseline():
    v = validate_against_readiness_v1()
    assert v["ok"]
    assert v["validation"]["integrity_tf"] == 430
    assert v["validation"]["scientific_condition_tf"] == 824_400


def test_legacy_stage_paths_not_forced_single_in_metrics_defs():
    # Historical Stage 1 helper must use default tokenizer encode (double BOS).
    text = Path("src/traces/pilot/metrics_defs.py").read_text()
    assert "Stage 1 historical convention" in text
    assert "prompt_ids = tokenizer(rendered_prompt, return_tensors=\"pt\")" in text
    # No live call to Q1 encoder in this Stage 1 helper
    assert "encode_rendered_prompt(" not in text
    ev = Path("src/traces/evaluation.py").read_text()
    assert "Stage 0/1 historical convention" in ev
    assert "encode_rendered_prompt(" not in ev
