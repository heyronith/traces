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


def test_r6_h1_three_states_and_unresolved_annotation():
    """R6: three decision states only; unresolved is annotation; marg entirely below → fail."""
    # Decisive marginal fail (entire interval < 0.1) with rank+share pass → H1-fail + annotation
    h_fail = evaluate_h1_direction(
        route_class="R-moved",
        direction="forward",
        rank_iv=Interval(1, 2, 1),
        share_iv=Interval(0.55, 0.9, 0.7),
        marginal_h15_iv=Interval(0.0, 0.05, 0.02),
        marginal_h16_iv=Interval(0.15, 0.4, 0.2),
        route_denom_ok=True,
    )
    assert h_fail["state"] == "H1-fail"
    assert h_fail.get("unresolved_marginal_contribution") is True

    # Rank+share pass, marginal straddles 0.1 → H1-indeterminate + annotation
    h_indet = evaluate_h1_direction(
        route_class="R-moved",
        direction="forward",
        rank_iv=Interval(1, 2, 1),
        share_iv=Interval(0.55, 0.9, 0.7),
        marginal_h15_iv=Interval(0.05, 0.15, 0.10),
        marginal_h16_iv=Interval(0.15, 0.4, 0.2),
        route_denom_ok=True,
    )
    assert h_indet["state"] == "H1-indeterminate"
    assert h_indet.get("unresolved_marginal_contribution") is True
    assert h_indet["state"] != "H1-unresolved-marginal-contribution"

    # Full pass
    h_pass = evaluate_h1_direction(
        route_class="R-moved",
        direction="reverse",
        rank_iv=Interval(1, 2, 1),
        share_iv=Interval(0.55, 0.9, 0.7),
        marginal_h15_iv=Interval(0.15, 0.4, 0.2),
        marginal_h16_iv=Interval(0.15, 0.4, 0.2),
        route_denom_ok=True,
    )
    assert h_pass["state"] == "H1-pass"
    assert h_pass.get("unresolved_marginal_contribution") is False

    # Rank decisive fail
    h_rank = evaluate_h1_direction(
        route_class="R-moved",
        direction="forward",
        rank_iv=Interval(4, 10, 5),
        share_iv=Interval(0.55, 0.9, 0.7),
        marginal_h15_iv=Interval(0.15, 0.4, 0.2),
        marginal_h16_iv=Interval(0.15, 0.4, 0.2),
        route_denom_ok=True,
    )
    assert h_rank["state"] == "H1-fail"


def test_r3_recipient_verification():
    # Expected IDs constructed independently (not live.tolist())
    expected_t = [2, 106, 154657, 10, 11]
    expected_c = [2, 106, 152798, 10, 11]
    al = derive_tag_alignment(list(expected_t), list(expected_c))
    good = torch.tensor([list(expected_t)])
    verify_recipient_input_ids(
        good, al, expected_tag="T", expected_full_ids=list(expected_t)
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_recipient_input_ids(
            torch.tensor([list(expected_c)]),
            al,
            expected_tag="T",
            expected_full_ids=list(expected_t),
        )
    # Non-tag token corrupted; tag ID retained — still reject
    bad = torch.tensor([list(expected_t)])
    bad[0, 0] = 99
    assert int(bad[0, al.tag_index].item()) == expected_t[al.tag_index]
    with pytest.raises(RuntimeError, match="mismatch"):
        verify_recipient_input_ids(
            bad, al, expected_tag="T", expected_full_ids=list(expected_t)
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


def test_f1_exact_reference_hash_pinned_numpy_213():
    """Exact F1 SHA under frozen NumPy 2.1.3 — never treat a skip as verification."""
    import hashlib
    import subprocess
    import sys
    import numpy as np

    pin_py = Path("results/q1_final_cpu_checks/.venv_tf4513/bin/python")
    record = Path("results/q1_implementation_prep/f1_draw_stream_hash_numpy213.txt")

    def _hash_now() -> str:
        return hashlib.sha256(draw_f1_resamples(180, 90, 90).full_set.tobytes()).hexdigest()

    if np.__version__.startswith("2.1.3"):
        h = _hash_now()
        assert h == F1_REF_HASH
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(
            f"interpreter {sys.executable}\nnumpy {np.__version__}\nsha256 {h}\n"
        )
        return

    # Host is not 2.1.3: execute the assertion inside the pinned venv (fail if missing)
    assert pin_py.is_file(), (
        f"Pinned NumPy 2.1.3 interpreter required at {pin_py}; "
        f"host numpy={np.__version__} cannot verify F1 reference alone"
    )
    script = (
        "import hashlib, sys; sys.path.insert(0, 'src'); "
        "import numpy as np; "
        "from traces.q1.stats_f1_f2 import draw_f1_resamples; "
        "h=hashlib.sha256(draw_f1_resamples(180,90,90).full_set.tobytes()).hexdigest(); "
        f"assert np.__version__.startswith('2.1.3'), np.__version__; "
        f"assert h=='{F1_REF_HASH}', h; "
        "print('interpreter', sys.executable); "
        "print('numpy', np.__version__); "
        "print('sha256', h)"
    )
    proc = subprocess.run(
        [str(pin_py), "-c", script],
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"Pinned F1 hash verification failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert F1_REF_HASH in proc.stdout
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(proc.stdout if proc.stdout.endswith("\n") else proc.stdout + "\n")


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
    expl = []
    for s in (42, 43, 44):
        eu = _make_unit(40, seed=s)
        eu.role = "exploratory_dev"
        eu.family = "l1523"
        eu.model_unit = f"exploratory_l1523_seed{s}"
        expl.append(eu)
    primaries = []
    for s in (42, 43, 44, 45, 46):
        pu = _make_unit(40, seed=s)
        pu.model_unit = f"primary_l19_seed{s}"
        primaries.append(pu)
    study = run_study(
        expl + primaries,
        held_out,
        exploratory_single_bos_ranks=[1, 2, 3],
    )
    assert study["exploratory_stop"]["triggered"] is False
    assert study["exploratory_stop"]["confirmatory_blocked"] is False
    assert study["study_level"] is not None


def test_workload_v1_3_totals():
    v = validate_against_readiness_v1()
    assert v["ok"]
    assert v["validation"]["scientific_condition_tf"] == 824_400
    assert v["validation"]["gen_passes"] == 3_600
    assert v["validation"]["integrity_tf"] == 430
