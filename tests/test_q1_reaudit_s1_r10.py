"""Targeted re-audit corrections: S1/S2, R3–R8, R10 (no model forwards)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from traces.q1.analysis import (
    Interval,
    classify_route,
    combine_h1_rmoved,
    evaluate_h1_direction,
    study_level_decision,
)
from traces.q1.driver import ModelUnitInput, run_model_unit_analysis, run_study
from traces.q1.injection import assert_tensor_compatible, make_residual_replace_hook
from traces.q1.margin import log_p_a, decision_margin
from traces.q1.patching import make_input_patch_hook
from traces.q1.recipient import verify_recipient_input_ids
from traces.q1.runtime_spec import (
    CUBLAS_WORKSPACE_CONFIG_VALUE,
    apply_determinism_settings,
    ensure_cublas_workspace_env,
)
from traces.q1.stats_f1_f2 import draw_f1_resamples
from traces.q1.tag_position import derive_tag_alignment
from traces.q1.workload import validate_against_readiness_v1

F1_REF_HASH = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def test_s1_indeterminate_from_classify_route_counts():
    # Build intervals that yield Indeterminate from classify_route
    n_iv = Interval(0.3, 0.7, 0.5)  # straddles
    s_iv = Interval(0.3, 0.7, 0.5)
    label = classify_route(n_iv, s_iv)
    assert label == "Indeterminate"
    labels = ["R-moved", label, "R-moved", "R-moved", "R-moved"]
    # Only 1 Indeterminate → not inconclusive
    assert study_level_decision(labels) == "SUPPORTED_WITHIN_FAMILY:R-moved"
    labels2 = ["R-moved", "Indeterminate", "Indeterminate", "R-moved", "R-moved"]
    assert study_level_decision(labels2) == "INCONCLUSIVE"


def test_s2_combine_pass_plus_indeterminate():
    fwd = {"state": "H1-pass", "direction": "forward", "share": "strong"}
    rev = {"state": "H1-indeterminate", "direction": "reverse"}
    out = combine_h1_rmoved(fwd, rev)
    assert out["full_H1"] == "H1-indeterminate"
    assert out["directional_partial_evidence"]["direction"] == "forward"
    both = combine_h1_rmoved(fwd, {"state": "H1-pass", "direction": "reverse"})
    assert both["full_H1"] == "H1-pass"


def test_r6_unresolved_marginal_descriptor():
    h = evaluate_h1_direction(
        route_class="R-moved",
        direction="forward",
        rank_iv=Interval(1, 2, 1),
        share_iv=Interval(0.55, 0.9, 0.7),
        marginal_h15_iv=Interval(0.0, 0.05, 0.02),  # fails marginal
        marginal_h16_iv=Interval(0.15, 0.4, 0.2),
        route_denom_ok=True,
    )
    assert h["state"] == "H1-unresolved-marginal-contribution"


def test_r3_recipient_verification():
    al = derive_tag_alignment([2, 106, 154657, 10, 11], [2, 106, 152798, 10, 11])
    good = torch.tensor([[2, 106, 154657, 10, 11]])
    verify_recipient_input_ids(
        good, al, expected_tag="T", expected_full_ids=[2, 106, 154657, 10, 11]
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_recipient_input_ids(
            torch.tensor([[2, 106, 152798, 10, 11]]),
            al,
            expected_tag="T",
            expected_full_ids=[2, 106, 154657, 10, 11],
        )


def test_r4_no_silent_cast():
    live = torch.zeros(1, 4, 8, dtype=torch.float32)
    bad = torch.zeros(1, 4, 8, dtype=torch.float16)
    with pytest.raises(RuntimeError, match="dtype"):
        assert_tensor_compatible(live, bad, context="test")
    hook = make_residual_replace_hook(bad)
    with pytest.raises(RuntimeError, match="dtype"):
        hook(nn.Identity(), (live,), {})


def test_r5_kwargs_and_args_conventions():
    rep = torch.ones(1, 3, 4)
    hook = make_residual_replace_hook(rep)
    # kwargs path
    args, kwargs = hook(nn.Identity(), tuple(), {"hidden_states": torch.ones(1, 3, 4), "extra": 1})
    assert torch.equal(kwargs["hidden_states"], rep)
    assert kwargs["extra"] == 1
    # args path
    args2, kwargs2 = hook(nn.Identity(), (torch.ones(1, 3, 4), "keep"), {})
    assert torch.equal(args2[0], rep)
    assert args2[1] == "keep"
    # patch hook kwargs
    donor = torch.zeros(1, 3, 32)
    donor[..., 1, 0:4] = 7
    ph = make_input_patch_hook(donor, slice(0, 4), (1,))
    live = torch.zeros(1, 3, 32)
    a, k = ph(nn.Identity(), tuple(), {"hidden_states": live})
    assert float(k["hidden_states"][0, 1, 0]) == 7.0


def test_r6_log_p_a_populated():
    logits = torch.tensor([0.0, 2.0, 0.5])
    lp = log_p_a(logits, payload_a=1)
    m = decision_margin(logits, b_i=2, payload_a=1)
    assert isinstance(lp, float)
    assert m == pytest.approx(lp - float(torch.log_softmax(logits.float(), dim=-1)[2]))


def test_r8_determinism_and_cublas():
    ensure_cublas_workspace_env(overwrite=True)
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG_VALUE
    apply_determinism_settings()
    with pytest.raises(RuntimeError):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = "bad"
        try:
            ensure_cublas_workspace_env(overwrite=False)
        finally:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG_VALUE


def test_f1_exact_reference_hash_requires_numpy_213():
    """Exact F1 reference: fe464187… under numpy 2.1.3 (no repeatability-only substitute)."""
    import hashlib
    import numpy as np

    h = hashlib.sha256(draw_f1_resamples(180, 90, 90).full_set.tobytes()).hexdigest()
    pin = Path("results/q1_implementation_prep/f1_draw_stream_hash_numpy213.txt")
    if pin.exists():
        recorded = pin.read_text().strip().split()[-1]
        assert recorded == F1_REF_HASH, "Pinned venv artifact must store exact F1 hash"
    if not np.__version__.startswith("2.1.3"):
        pytest.skip(
            f"Exact F1 hash assert requires numpy 2.1.3; got {np.__version__} "
            f"(computed={h}; pin={F1_REF_HASH})"
        )
    assert h == F1_REF_HASH


def test_r7_post_load_rejects_mixed_dtypes_and_echo():
    from traces.q1.post_load import verify_post_load

    class _Cfg:
        _attn_implementation = "eager"
        # no _commit_hash → cannot echo-verify

    class _P:
        def __init__(self, dtype):
            self.dtype = dtype
            self.requires_grad = False

    class _M:
        config = _Cfg()
        peft_config = {"default": type("PC", (), {"target_modules": ["q_proj"]})()}

        def parameters(self):
            yield _P(torch.bfloat16)
            yield _P(torch.float32)

        def named_parameters(self):
            yield ("layers.0.self_attn.q_proj.lora_A.default.weight", _P(torch.float32))

    class _Tok:
        is_fast = True

    rep = verify_post_load(
        _M(),
        _Tok(),
        expected_base_revision="abc",
        expected_adapter_revision="def",
        require_eager=True,
        require_bf16=True,
    )
    assert rep.ok is False
    assert any("dtype" in f.lower() or "Disallowed" in f for f in rep.failures)
    assert any("revision" in f.lower() or "SHA" in f for f in rep.failures)


def _make_unit(n=40, seed=0) -> ModelUnitInput:
    rng = np.random.default_rng(seed)
    ids = [f"p{i}" for i in range(n)]
    m_CC = rng.normal(0, 0.01, n)
    m_TT = m_CC + 3.0
    m_CT = m_CC + 0.1
    m_TC = m_TT - 0.1
    # 64 pairs with core (6,5) strongest
    pair_fwd = {}
    pair_rev = {}
    for h15 in range(8):
        for h16 in range(8):
            base = rng.normal(0.2, 0.05, n)
            if (h15, h16) == (6, 5):
                base = base + 1.5
            pair_fwd[(h15, h16)] = base
            pair_rev[(h15, h16)] = base * 0.9
    head_fwd = {(L, h): rng.normal(0.1, 0.02, n) for L in range(0, 19) for h in range(8)}
    head_fwd[(15, 6)] = pair_fwd[(6, 5)] * 0.6
    head_fwd[(16, 5)] = pair_fwd[(6, 5)] * 0.55
    head_rev = {k: v * 0.9 for k, v in head_fwd.items()}
    e_route = m_TT - m_CT
    return ModelUnitInput(
        model_unit="l19/seed42",
        family="l19",
        role="primary_confirmatory",
        prompt_ids=ids,
        m_TT=m_TT,
        m_CT=m_CT,
        m_TC=m_TC,
        m_CC=m_CC,
        eligible_mask=np.ones(n, dtype=bool),
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.95,
        n_frozen=40,
        pair_effects_fwd=pair_fwd,
        pair_effects_rev=pair_rev,
        head_effects_fwd=head_fwd,
        head_effects_rev=head_rev,
        e_route_fwd=e_route,
        e_route_rev=m_TC - m_CC,
    )


def test_r10_driver_e2e_exact_keys():
    held = [f"p{i}" for i in range(200)]
    # Use first 40 as eligible subset ids matching unit
    unit = _make_unit(40)
    # Align held-out order: put unit ids at start
    held_out = unit.prompt_ids + [f"h{i}" for i in range(160)]
    res = run_model_unit_analysis(unit, held_out)
    d = res.to_dict()
    assert d["gates"]["passed"] is True or d["gates"]["passed"] is False
    assert "route_class" in d["route"]
    assert "full_H1" in d["h1"] or "forward" in d["h1"]
    assert "crossfit" in d
    study = run_study(
        [unit, unit, unit, unit, unit],
        held_out,
        exploratory_single_bos_ranks=[1, 2, 3],
    )
    assert study["exploratory_stop"]["triggered"] is False
    assert study["study_level"] is not None


def test_workload_v1_3_totals():
    v = validate_against_readiness_v1()
    assert v["ok"]
    assert v["validation"]["scientific_condition_tf"] == 824_400
    assert v["validation"]["gen_passes"] == 3_600
    assert v["validation"]["integrity_tf"] == 430
