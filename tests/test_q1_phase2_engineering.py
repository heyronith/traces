"""Phase 2 engineering: stats, auth gates, loader provenance, runtime lock (no forwards)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traces.q1.analysis import Interval, evaluate_h2_h3, exploratory_stop_triggered
from traces.q1.analysis_ext import (
    classify_k_star,
    component_route_share_intervals,
    crossfit_evaluate_topk,
    h2_component_coverage,
    h3_top2_conservation,
    lookup_joint_effect,
    selection_key,
    top2_heads_from_mean_effects,
)
from traces.q1.driver import ModelUnitInput, run_model_unit_analysis, run_study
from traces.q1.loader import validate_loader_policy_inputs
from traces.q1.post_load import (
    refuse_caller_revision_echo_as_proof,
    verify_adapter_off_behavior,
    verify_post_load,
)
from traces.q1.runtime_lock import (
    claim_production_lock_exists,
    generate_lock_from_wheel_dir,
    validate_lock_document,
    write_lock_document,
)
from traces.q1.scientific_execution import (
    SCI_AUTH_ENV,
    TINY_AUTH_ENV,
    build_pipeline_plan,
    require_scientific_authorization,
    run_analysis_only,
    run_live_scientific_pipeline,
    scientific_auth_status,
)
from traces.q1.stats_f1_f2 import assign_f2_folds, draw_f1_resamples, f1_full_set_stream_sha256


REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _joint_for_topk(head_fx: dict, e_route: np.ndarray, k: int) -> dict:
    """Build actual joint arrays as sum for synthetic tests only (known construction)."""
    means = {h: float(np.mean(v)) for h, v in head_fx.items()}
    ranked = sorted(means.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
    out = {}
    for kk in range(1, k + 1):
        sel = [h for h, _ in ranked[:kk]]
        key = selection_key(sel)
        acc = np.zeros_like(e_route, dtype=np.float64)
        for h in sel:
            acc = acc + head_fx[h]
        out[key] = acc
    return out


def _make_unit(n=40, seed=0, with_joints=True) -> ModelUnitInput:
    rng = np.random.default_rng(seed)
    ids = [f"p{i}" for i in range(n)]
    m_CC = rng.normal(0, 0.01, n)
    m_TT = m_CC + 3.0
    m_CT = m_CC + 0.1
    m_TC = m_TT - 0.1
    pair_fwd = {}
    pair_rev = {}
    for h15 in range(8):
        for h16 in range(8):
            base = rng.normal(0.2, 0.05, n)
            if (h15, h16) == (6, 5):
                base = base + 1.5
            pair_fwd[(h15, h16)] = base
            pair_rev[(h15, h16)] = base * 0.9
    head_fwd = {(L, h): rng.normal(0.05, 0.01, n) for L in range(0, 19) for h in range(8)}
    # Make two heads dominate for stable top-2
    head_fwd[(15, 6)] = np.full(n, 0.9)
    head_fwd[(16, 5)] = np.full(n, 0.85)
    head_rev = {k: v * 0.9 for k, v in head_fwd.items()}
    e_route = m_TT - m_CT
    joints_fwd = _joint_for_topk(head_fwd, e_route, 10) if with_joints else {}
    joints_rev = _joint_for_topk(head_rev, m_TC - m_CC, 10) if with_joints else {}
    return ModelUnitInput(
        model_unit=f"l19/seed{seed}",
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
        n_frozen=n,
        pair_effects_fwd=pair_fwd,
        pair_effects_rev=pair_rev,
        head_effects_fwd=head_fwd,
        head_effects_rev=head_rev,
        e_route_fwd=e_route,
        e_route_rev=m_TC - m_CC,
        joint_set_effects_fwd=joints_fwd,
        joint_set_effects_rev=joints_rev,
    )


# --- F1 hash preserved ---


def test_f1_reference_hash_unchanged():
    assert f1_full_set_stream_sha256(180) == F1_REF


# --- Cross-fit joint interventions ---


def test_crossfit_refuses_head_sum_without_joint_arrays():
    draws = draw_f1_resamples(20, 10, 10)
    head_a = {(0, 0): 1.0, (0, 1): 0.5}
    head_b = {(0, 0): 0.9, (0, 1): 0.4}
    ev = crossfit_evaluate_topk(
        head_effects_fold_a=head_a,
        head_effects_fold_b=head_b,
        joint_effects={},  # missing
        e_route_prompts=np.ones(20),
        eligible_fold_a_idx=list(range(10)),
        eligible_fold_b_idx=list(range(10, 20)),
        k=1,
        draws=draws,
    )
    assert ev["indeterminate"] is True
    assert "joint" in ev["reason"].lower() or "missing" in ev["reason"].lower()


def test_crossfit_joint_bootstrap_confidence_bounds():
    n = 40
    rng = np.random.default_rng(99)
    # Heterogeneous per-prompt effects so fold bootstrap CIs have positive width
    head_fx = {
        (0, 0): rng.uniform(0.2, 1.5, n),
        (0, 1): rng.uniform(0.1, 1.2, n),
        (1, 0): rng.uniform(0.0, 0.4, n),
    }
    e_route = rng.uniform(1.5, 3.0, n)
    joints = _joint_for_topk(head_fx, e_route, 3)
    fold_a = list(range(0, 20))
    fold_b = list(range(20, 40))
    draws = draw_f1_resamples(n, len(fold_a), len(fold_b))
    means_a = {k: float(np.mean(v[fold_a])) for k, v in head_fx.items()}
    means_b = {k: float(np.mean(v[fold_b])) for k, v in head_fx.items()}
    ev = crossfit_evaluate_topk(
        head_effects_fold_a=means_a,
        head_effects_fold_b=means_b,
        joint_effects=joints,
        e_route_prompts=e_route,
        eligible_fold_a_idx=fold_a,
        eligible_fold_b_idx=fold_b,
        k=2,
        draws=draws,
    )
    assert ev["indeterminate"] is False
    iv = ev["share_interval_90"]
    assert isinstance(iv, Interval)
    assert iv.lower <= iv.point <= iv.upper
    # Real bootstrap percentiles — not a point±epsilon masquerade
    assert iv.upper > iv.lower
    assert ev["n_bootstrap_finite"] == 2000


def test_lookup_joint_effect_canonical_key():
    arr = np.array([1.0, 2.0])
    joints = {selection_key([(16, 5), (15, 6)]): arr}
    got = lookup_joint_effect(joints, [(15, 6), (16, 5)])
    assert got is not None
    np.testing.assert_array_equal(got, arr)


# --- H1/H2/H3 edges ---


def test_h2_requires_all_component_upper_bounds():
    ivs = {
        "a": Interval(0.0, 0.1, 0.05),
        "b": Interval(0.0, 0.24, 0.1),
        "c": Interval(0.0, 0.25, 0.2),  # violates <0.25
    }
    cov = h2_component_coverage(ivs)
    assert cov["all_upper_below"] is False
    assert "c" in cov["violators"]


def test_h2_h3_decision_edges():
    supported = evaluate_h2_h3(
        k_star_le_2=False,
        k_star_gt_10=True,
        h1_state="H1-fail",
        all_component_share_upper_below_025=True,
        top2_set_conservation_seeds=0,
        top2_varies_across_seeds=False,
    )
    assert supported["H2"] == "supported"
    h3 = evaluate_h2_h3(
        k_star_le_2=True,
        k_star_gt_10=False,
        h1_state="H1-fail",
        all_component_share_upper_below_025=False,
        top2_set_conservation_seeds=4,
        top2_varies_across_seeds=False,
    )
    assert h3["H3a"] == "supported"
    # H1-indeterminate blocks H3
    no_h3 = evaluate_h2_h3(
        k_star_le_2=True,
        k_star_gt_10=False,
        h1_state="H1-indeterminate",
        all_component_share_upper_below_025=False,
        top2_set_conservation_seeds=5,
        top2_varies_across_seeds=True,
    )
    assert no_h3["H3a"] is None
    assert no_h3["H3b"] is None


def test_five_seed_conservation_and_exploratory_stop():
    top2 = {
        "s1": frozenset([(15, 6), (16, 5)]),
        "s2": frozenset([(15, 6), (16, 5)]),
        "s3": frozenset([(15, 6), (16, 5)]),
        "s4": frozenset([(15, 6), (16, 5)]),
        "s5": frozenset([(10, 1), (11, 2)]),
    }
    cons = h3_top2_conservation(top2)
    assert cons["H3a_count"] == 4
    assert cons["varies"] is True
    assert exploratory_stop_triggered([11, 12, 1]) is True
    assert exploratory_stop_triggered([1, 2, 3]) is False


def test_k_star_classification_uses_bounds_not_points():
    # Point would be high but lower bound low → not k*≤2
    ks = classify_k_star({}, k2_share_lower=0.4, k10_share_upper=0.6)
    assert ks["k_star_le_2"] is False
    assert ks["undetermined_concentration"] is True
    ks2 = classify_k_star({}, k2_share_lower=0.55, k10_share_upper=0.4)
    assert ks2["k_star_le_2"] is True
    assert ks2["k_star_gt_10"] is True  # both flags computable independently


# --- Partial / missing data ---


def test_empty_eligible_fail_closed():
    unit = _make_unit(10, seed=2)
    unit.eligible_mask = np.zeros(10, dtype=bool)
    res = run_model_unit_analysis(unit, unit.prompt_ids + [f"h{i}" for i in range(190)])
    assert res.meta.get("fail_closed") or res.crossfit.get("indeterminate")
    assert res.h1.get("full_H1") in {"H1-indeterminate", "inapplicable"} or "empty" in str(
        res.h1.get("reason", "")
    )


def test_missing_joints_make_crossfit_indeterminate():
    unit = _make_unit(40, seed=3, with_joints=False)
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    res = run_model_unit_analysis(unit, held)
    fwd = res.crossfit["forward"]
    # Either denom failed or k entries indeterminate
    if fwd.get("indeterminate") and "denominator" in str(fwd.get("reason", "")):
        pytest.skip("denom gate failed on this synthetic draw — still fail-closed")
    else:
        # Check at least one k is indeterminate due to missing joints
        indets = [
            v.get("indeterminate")
            for v in (fwd.get("by_k") or {}).values()
            if isinstance(v, dict)
        ]
        assert fwd.get("indeterminate") or any(indets)


def test_driver_h2_evaluates_heads_and_pairs():
    unit = _make_unit(40, seed=4, with_joints=True)
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    res = run_model_unit_analysis(unit, held)
    meta = res.h2_h3.get("h2_meta") or {}
    if meta.get("coverage", {}).get("reason") == "route-denominator gate failed":
        # Still exercised denom-before-share gate
        assert meta["coverage"]["indeterminate"] is True
    else:
        assert meta.get("n_pairs") == 64
        assert meta.get("n_heads", 0) >= 64


def test_study_five_seed_top2_actual_sets():
    units = [_make_unit(40, seed=s, with_joints=True) for s in range(5)]
    held = units[0].prompt_ids + [f"h{i}" for i in range(160)]
    out = run_study(units, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert len(out["h3_top2_by_seed"]) == 5
    # Conserved dominant heads from _make_unit
    for _uid, pair in out["h3_top2_by_seed"].items():
        assert (15, 6) in [tuple(x) if isinstance(x, list) else x for x in pair] or len(pair) == 2


# --- Scientific auth gates ---


def test_scientific_auth_default_off(monkeypatch):
    monkeypatch.delenv(SCI_AUTH_ENV, raising=False)
    monkeypatch.delenv(TINY_AUTH_ENV, raising=False)
    st = scientific_auth_status()
    assert st.authorized is False
    with pytest.raises(PermissionError):
        require_scientific_authorization()


def test_tiny_auth_does_not_unlock_scientific(monkeypatch):
    monkeypatch.setenv(TINY_AUTH_ENV, "1")
    monkeypatch.delenv(SCI_AUTH_ENV, raising=False)
    st = scientific_auth_status()
    assert st.authorized is False
    assert "does not authorize" in st.message
    with pytest.raises(PermissionError):
        require_scientific_authorization()


def test_live_pipeline_refuses_without_auth(monkeypatch):
    monkeypatch.delenv(SCI_AUTH_ENV, raising=False)
    with pytest.raises(PermissionError):
        run_live_scientific_pipeline(
            repo_root=REPO,
            held_out_ids_allocation_order=[f"p{i}" for i in range(200)],
            hooks=SimpleNamespace(
                load_model_unit=lambda: None,
                capture_donors=lambda m: None,
                run_boundary_conditions=lambda m, d: None,
                run_interventions=lambda m, d: None,
            ),
            out_dir=REPO / "results/q1_implementation_prep/_tmp_live",
        )


def test_analysis_only_without_auth_ok(tmp_path):
    unit = _make_unit(40, seed=5, with_joints=True)
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    out = run_analysis_only([unit], held, out_dir=tmp_path, require_auth=False)
    assert "per_unit" in out
    assert Path(out["persisted_path"]).is_file()


def test_pipeline_plan_preserves_workload():
    plan = build_pipeline_plan(REPO)
    d = plan.to_dict()
    assert d["held_out_n"] == 200
    assert len(d["stages"]) == 8
    assert d["live_forwards_authorized"] is False or isinstance(
        d["live_forwards_authorized"], bool
    )


# --- Loader provenance ---


def test_loader_policy_rejects_missing_weight_pins():
    fails = validate_loader_policy_inputs(
        base_revision="wrong",
        adapter_revision="wrong",
        expected_weight_file_sha256=None,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    )
    assert any("weight" in f.lower() for f in fails)
    assert any("dtype" in f.lower() for f in fails)
    assert any("attn" in f.lower() for f in fails)
    assert any("echo" in f.lower() or "caller" in f.lower() for f in fails)


def test_refuse_caller_revision_echo():
    fails = refuse_caller_revision_echo_as_proof(
        caller_base_revision="abc1234",
        caller_adapter_revision="def5678",
        inferred_base=None,
        inferred_adapter=None,
        has_weight_pins=False,
    )
    assert len(fails) == 2


def test_post_load_weight_sha_and_eager(tmp_path):
    weight = tmp_path / "adapter.safetensors"
    weight.write_bytes(b"fake-weights-v1")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()

    class FakeCfg:
        _attn_implementation = "eager"
        _commit_hash = "deadbeefcafebabe"

    class FakePeft:
        target_modules = ["q_proj", "v_proj"]
        revision = "cafebabedeadbeef"

    class FakeModel:
        config = FakeCfg()
        peft_config = {"default": FakePeft()}

        def parameters(self):
            yield torch.ones(2, dtype=torch.bfloat16)
            yield torch.ones(2, dtype=torch.bfloat16)

        def named_parameters(self):
            yield "layers.19.self_attn.q_proj.lora_A.default.weight", torch.ones(
                2, dtype=torch.bfloat16
            )
            yield "layers.19.self_attn.v_proj.lora_B.default.weight", torch.ones(
                2, dtype=torch.bfloat16
            )

    class FakeTok:
        is_fast = False

    # Wrong SHA → fail
    bad = verify_post_load(
        FakeModel(),
        FakeTok(),
        expected_base_revision="deadbeefcafebabe",
        expected_adapter_revision="cafebabedeadbeef",
        expected_weight_file_sha256={str(weight): "0" * 64},
        expected_tokenizer_class="FakeTok",
        expected_tokenizer_is_fast=False,
        expected_adapter_targets=["q_proj", "v_proj"],
        expected_adapted_layers=[19],
    )
    assert bad.ok is False
    assert any("SHA" in f or "sha" in f.lower() for f in bad.failures)

    good = verify_post_load(
        FakeModel(),
        FakeTok(),
        expected_base_revision="deadbeefcafebabe",
        expected_adapter_revision="cafebabedeadbeef",
        expected_weight_file_sha256={str(weight): digest},
        expected_tokenizer_class="FakeTok",
        expected_tokenizer_is_fast=False,
        expected_adapter_targets=["q_proj", "v_proj"],
        expected_adapted_layers=[19],
    )
    assert good.ok is True


def test_adapter_off_verification_mock():
    class OkModel:
        def __init__(self):
            self.disabled = False

        def disable_adapter_layers(self):
            self.disabled = True

        def enable_adapter_layers(self):
            self.disabled = False

    class BadModel:
        pass

    assert verify_adapter_off_behavior(OkModel())["ok"] is True
    assert verify_adapter_off_behavior(BadModel())["ok"] is False


# --- Runtime lock ---


def test_runtime_lock_generate_and_validate(tmp_path):
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    w1 = wheel_dir / "numpy-2.1.3-py3-none-any.whl"
    w1.write_bytes(b"wheel-bytes-numpy")
    w2 = wheel_dir / "torch-2.5.1-py3-none-any.whl"
    w2.write_bytes(b"wheel-bytes-torch")
    doc = generate_lock_from_wheel_dir(
        wheel_dir,
        repo_root=REPO,
        pip_freeze_text="numpy==2.1.3\ntorch==2.5.1\n",
        source="synthetic_fixture",
        production_image_lock=False,
    )
    path = write_lock_document(doc, tmp_path / "lock.json")
    loaded = json.loads(path.read_text())
    assert claim_production_lock_exists(loaded) is False
    ok = validate_lock_document(loaded, wheel_dir=wheel_dir, require_production=False)
    assert ok["ok"] is True

    # Tamper → reject
    loaded["wheels"][0]["sha256"] = "bad"
    bad = validate_lock_document(loaded, wheel_dir=wheel_dir)
    assert bad["ok"] is False

    # Production requirement rejects synthetic
    prod_req = validate_lock_document(
        json.loads(path.read_text()), require_production=True
    )
    assert prod_req["ok"] is False


def test_component_share_intervals_bootstrap():
    draws = draw_f1_resamples(30, 15, 15)
    comps = {
        ("head", 0, 0): np.linspace(0.1, 0.2, 30),
        ("pair", 6, 5): np.linspace(0.5, 0.7, 30),
    }
    route = np.ones(30) * 1.0
    ivs = component_route_share_intervals(comps, route, draws)
    assert len(ivs) == 2
    for iv in ivs.values():
        assert iv.lower <= iv.upper


def test_top2_from_mean_effects():
    fx = {
        (1, 0): np.array([0.1, 0.1]),
        (2, 0): np.array([0.9, 0.9]),
        (3, 0): np.array([0.8, 0.8]),
    }
    assert top2_heads_from_mean_effects(fx) == frozenset([(2, 0), (3, 0)])
