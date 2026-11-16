"""Phase 2 internal audit corrections C1–C6 (synthetic CPU only)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from traces.q1.analysis import Interval, evaluate_h2_h3
from traces.q1.analysis_ext import (
    classify_k_star,
    crossfit_evaluate_topk,
    h2_component_coverage,
    h3_top2_conservation,
)
from traces.q1.component_sets import (
    EXPECTED_L15_L16_PAIRS,
    EXPECTED_UPSTREAM_HEADS,
    validate_head_effect_map,
    validate_pair_effect_map,
)
from traces.q1.driver import (
    ADAPTER_OFF_ROLE,
    EXPLORATORY_ROLE,
    PRIMARY_ROLE,
    SECONDARY_ROLE,
    ModelUnitInput,
    run_study,
)
from traces.q1.loader import validate_loader_policy_inputs
from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
from traces.q1.post_load import (
    validate_weight_manifest_roles,
    verify_adapter_off_behavior,
    verify_post_load,
)
from traces.q1.runtime_lock import (
    claim_production_lock_exists,
    generate_lock_from_wheel_dir,
    write_lock_document,
)
from traces.q1.runtime_spec import REQUIRED_TORCH_DTYPE
from traces.q1.scientific_execution import (
    SCI_AUTH_ENV,
    ConditionOutputBundle,
    ScientificPipelineExecutor,
    run_live_scientific_pipeline,
    run_scientific_preflight,
    run_typed_pipeline_from_bundle,
    scientific_auth_status,
)
from traces.q1.stats_f1_f2 import draw_f1_resamples, f1_full_set_stream_sha256

REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def test_f1_hash_still_frozen():
    assert f1_full_set_stream_sha256(180) == F1_REF


# --- C2 adversarial ---


def test_missing_k_cannot_support_concentration():
    ks = classify_k_star({}, k2_share_lower=None, k10_share_upper=None, k_scan_complete=False)
    assert ks["definitive_refused"] is True
    assert ks["k_star_le_2"] is False
    assert ks["k_star_gt_10"] is False


def test_wrong_length_joint_effect_indeterminate():
    draws = draw_f1_resamples(20, 10, 10)
    head_a = {(0, 0): 1.0}
    head_b = {(0, 0): 0.9}
    joints = {((0, 0),): np.ones(10)}  # wrong length vs n_eligible=20
    ev = crossfit_evaluate_topk(
        head_effects_fold_a=head_a,
        head_effects_fold_b=head_b,
        joint_effects=joints,
        e_route_prompts=np.ones(20),
        eligible_fold_a_idx=list(range(10)),
        eligible_fold_b_idx=list(range(10, 20)),
        k=1,
        draws=draws,
        n_eligible=20,
    )
    assert ev["indeterminate"] is True


def test_h2_missing_components_not_all_upper_below():
    # Only one head component — must not satisfy H2
    ivs = {("head", 0, 0): Interval(0.0, 0.1, 0.05)}
    cov = h2_component_coverage(ivs)
    assert cov["all_upper_below"] is False
    assert cov["indeterminate"] is True
    assert cov["missing_head_count"] > 0


def test_h2_nonfinite_upper_not_support():
    ivs = {
        ("head",) + h: Interval(0.0, 0.1, 0.05) for h in list(EXPECTED_UPSTREAM_HEADS)[:151]
    }
    # leave one missing; add nonfinite
    ivs[("head", 0, 0)] = Interval(float("nan"), float("nan"), float("nan"))
    for p in EXPECTED_L15_L16_PAIRS:
        ivs[("pair",) + p] = Interval(0.0, 0.1, 0.05)
    cov = h2_component_coverage(ivs)
    assert cov["all_upper_below"] is False


def test_h3_partial_seed_map_ineligible():
    partial = {
        "s1": frozenset([(15, 6), (16, 5)]),
        "s2": frozenset([(15, 6), (16, 5)]),
        "s3": frozenset([(15, 6), (16, 5)]),
    }
    cons = h3_top2_conservation(partial, require_five_seeds=True)
    assert cons["H3a_eligible"] is False
    assert cons["H3b_eligible"] is False
    h = evaluate_h2_h3(
        k_star_le_2=True,
        k_star_gt_10=False,
        h1_state="H1-fail",
        all_component_share_upper_below_025=False,
        top2_set_conservation_seeds=int(cons.get("H3a_count") or 0),
        top2_varies_across_seeds=False,
    )
    # Without eligible conservation, driver clears H3 — here evaluate_h2_h3 may still
    # set H3a if count>=4, but count is 0 when ineligible
    assert cons["H3a_count"] == 0


def test_expected_component_set_sizes():
    assert len(EXPECTED_UPSTREAM_HEADS) == 152
    assert len(EXPECTED_L15_L16_PAIRS) == 64


def test_head_map_rejects_incomplete_identities():
    n = 5
    incomplete = {(0, 0): np.ones(n), (0, 1): np.ones(n)}
    chk = validate_head_effect_map(incomplete, n_eligible=n)
    assert chk["ok"] is False
    pairs = {(0, 0): np.ones(n)}
    assert validate_pair_effect_map(pairs, n_eligible=n)["ok"] is False


# --- C3 study roles ---


def _tiny_unit(role: str, seed: int, n: int = 20) -> ModelUnitInput:
    rng = np.random.default_rng(seed)
    ids = [f"{role}-{seed}-p{i}" for i in range(n)]
    m_CC = rng.normal(0, 0.01, n)
    m_TT = m_CC + 2.0
    pair = {(h15, h16): rng.normal(0.1, 0.01, n) for h15 in range(8) for h16 in range(8)}
    pair[(6, 5)] = pair[(6, 5)] + 1.0
    heads = {(L, h): rng.normal(0.05, 0.01, n) for L in range(19) for h in range(8)}
    heads[(15, 6)] = np.full(n, 0.5)
    heads[(16, 5)] = np.full(n, 0.4)
    return ModelUnitInput(
        model_unit=f"{role}/seed{seed}",
        family="l19" if "primary" in role or role == PRIMARY_ROLE else "l1523",
        role=role,
        prompt_ids=ids,
        m_TT=m_TT,
        m_CT=m_CC + 0.2,
        m_TC=m_TT - 0.2,
        m_CC=m_CC,
        eligible_mask=np.ones(n, dtype=bool),
        trigger_asr=0.01 if role == ADAPTER_OFF_ROLE else 0.95,
        clean_false_fire=0.0,
        first_token_validity=0.95,
        n_frozen=n,
        pair_effects_fwd=pair,
        pair_effects_rev={k: v * 0.9 for k, v in pair.items()},
        head_effects_fwd=heads,
        head_effects_rev={k: v * 0.9 for k, v in heads.items()},
        e_route_fwd=m_TT - (m_CC + 0.2),
        e_route_rev=(m_TT - 0.2) - m_CC,
    )


def test_exploratory_stop_blocks_confirmatory():
    units = []
    for s in (42, 43, 44):
        u = _tiny_unit(EXPLORATORY_ROLE, s, 50)
        u.family = "l1523"
        u.model_unit = f"exploratory_l1523_seed{s}"
        units.append(u)
    units.append(_tiny_unit(PRIMARY_ROLE, 42, 40))
    held = [f"h{i}" for i in range(200)]
    out = run_study(units, held, exploratory_single_bos_ranks=[11, 12, 1])
    assert out["exploratory_stop"]["triggered"] is True
    assert out["exploratory_stop"]["confirmatory_blocked"] is True
    assert out["study_level"] is None
    assert any(p.get("skipped") for p in out["per_unit"] if p.get("role") == PRIMARY_ROLE)


def test_study_runs_all_roles_when_stop_clear():
    def _u(role, seed, n=40, family=None):
        u = _tiny_unit(role, seed, n)
        # Force frozen exploratory naming/family for D4
        if role == EXPLORATORY_ROLE:
            u.family = "l1523"
            u.model_unit = f"exploratory_l1523_seed{seed}"
        elif family:
            u.family = family
        return u

    units = (
        [_u(EXPLORATORY_ROLE, s, 50) for s in (42, 43, 44)]
        + [_u(PRIMARY_ROLE, s, 40, family="l19") for s in (42, 43, 44, 45, 46)]
        + [_u(SECONDARY_ROLE, s, 40, family="l1523") for s in (45, 46)]
        + [_u(ADAPTER_OFF_ROLE, 0, 40, family="base")]
    )
    held = units[3].prompt_ids + [f"x{i}" for i in range(160)]
    out = run_study(units, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["exploratory_stop"]["confirmatory_blocked"] is False
    roles = out["roles_executed"]
    assert roles[EXPLORATORY_ROLE] == 3
    assert roles[PRIMARY_ROLE] == 5
    assert roles[SECONDARY_ROLE] == 2
    assert roles[ADAPTER_OFF_ROLE] == 1


# --- C1 typed pipeline ---


def test_typed_pipeline_assembles_model_unit(tmp_path):
    n = 20
    u = _tiny_unit(PRIMARY_ROLE, 1, n)
    bundle = ConditionOutputBundle(
        model_unit=u.model_unit,
        family=u.family,
        role=u.role,
        prompt_ids=u.prompt_ids,
        m_TT=u.m_TT,
        m_CT=u.m_CT,
        m_TC=u.m_TC,
        m_CC=u.m_CC,
        eligible_mask=u.eligible_mask,
        trigger_asr=u.trigger_asr,
        clean_false_fire=u.clean_false_fire,
        first_token_validity=u.first_token_validity,
        n_frozen=u.n_frozen,
        pair_effects_fwd=u.pair_effects_fwd,
        pair_effects_rev=u.pair_effects_rev,
        head_effects_fwd=u.head_effects_fwd,
        head_effects_rev=u.head_effects_rev,
        e_route_fwd=u.e_route_fwd,
        e_route_rev=u.e_route_rev,
        joint_set_effects_fwd={((15, 6),): u.head_effects_fwd[(15, 6)]},
        joint_set_effects_rev={((15, 6),): u.head_effects_rev[(15, 6)]},
        catalog_condition_ids=["single_head|forward"],
        eligible_prompt_ids=list(u.prompt_ids),
    )
    provenance = {
        "base_revision": PINNED_BASE_REVISION,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "weight_file_sha256": {
            "base:/resolved/base_model.safetensors": "a" * 64,
            "adapter:/resolved/adapter_model.safetensors": "b" * 64,
        },
        "attn_implementation": "eager",
        "torch_dtype": "torch.bfloat16",
    }
    donors = {
        "n_donor_caches": 20,
        "clean_references_frozen": True,
        "instruction_ids": u.prompt_ids,
    }
    held = u.prompt_ids + [f"h{i}" for i in range(180)]
    out = run_typed_pipeline_from_bundle(
        repo_root=REPO,
        out_dir=tmp_path,
        held_out_ids=held,
        bundles=[bundle],
        provenance=provenance,
        donor_evidence=donors,
        exploratory_ranks=[1, 2, 3],
        skip_live_preflight=True,
    )
    assert out["all_complete"] is True
    assert all(s["status"] == "complete" for s in out["stages"].values())


def test_opaque_callback_does_not_complete_stage5(tmp_path):
    ex = ScientificPipelineExecutor(REPO, tmp_path)
    st = ex.stage5_interventions({"ok": True})  # type: ignore[arg-type]
    assert st.status == "failed"


def test_preflight_fail_closed_without_auth_f5_lock():
    pre = run_scientific_preflight(REPO, require_sci_auth=True)
    assert pre.ok is False
    assert any("authoriz" in f.lower() or "SCI" in f or "unset" in f.lower() for f in pre.failures)
    assert any("F5" in f for f in pre.failures)
    assert any("lock" in f.lower() for f in pre.failures)


def test_live_still_disabled(monkeypatch):
    monkeypatch.delenv(SCI_AUTH_ENV, raising=False)
    assert scientific_auth_status().authorized is False
    with pytest.raises(PermissionError):
        from traces.q1.scientific_execution import LiveForwardHooks

        run_live_scientific_pipeline(
            repo_root=REPO,
            held_out_ids_allocation_order=[f"p{i}" for i in range(200)],
            hooks=LiveForwardHooks(),
            out_dir=REPO / "results/q1_implementation_prep/_tmp",
        )


# --- C4 loader ---


def test_absent_adapted_layers_rejected(tmp_path):
    weight = tmp_path / "adapter_model.safetensors"
    weight.write_bytes(b"x")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()

    class FakeCfg:
        _attn_implementation = "eager"
        _commit_hash = PINNED_BASE_REVISION

    class FakePeft:
        target_modules = ["q_proj"]
        revision = PINNED_ADAPTER_REVISION

    class FakeModel:
        config = FakeCfg()
        peft_config = {"default": FakePeft()}

        def parameters(self):
            yield torch.ones(2, dtype=torch.bfloat16)

        def named_parameters(self):
            # No lora layer params → empty adapted inventory
            return iter([])

    class FakeTok:
        is_fast = False

    rep = verify_post_load(
        FakeModel(),
        FakeTok(),
        expected_base_revision=PINNED_BASE_REVISION,
        expected_adapter_revision=PINNED_ADAPTER_REVISION,
        expected_weight_file_sha256={str(weight): digest},
        expected_adapter_targets=["q_proj"],
        expected_adapted_layers=[19, 20, 21, 22, 23, 24, 25],
        expected_tokenizer_class="FakeTok",
    )
    assert rep.ok is False
    assert any("adapted" in f.lower() for f in rep.failures)


def test_revision_prefix_match_rejected(tmp_path):
    weight = tmp_path / "adapter_model.safetensors"
    weight.write_bytes(b"y")
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()

    class FakeCfg:
        _attn_implementation = "eager"
        _commit_hash = PINNED_BASE_REVISION[:7]  # prefix only

    class FakePeft:
        target_modules = ["q_proj"]
        revision = PINNED_ADAPTER_REVISION[:7]

    class FakeModel:
        config = FakeCfg()
        peft_config = {"default": FakePeft()}

        def parameters(self):
            yield torch.ones(2, dtype=torch.bfloat16)

        def named_parameters(self):
            yield "layers.19.self_attn.q_proj.lora_A.weight", torch.ones(
                2, dtype=torch.bfloat16
            )

    class FakeTok:
        is_fast = False

    rep = verify_post_load(
        FakeModel(),
        FakeTok(),
        expected_base_revision=PINNED_BASE_REVISION,
        expected_adapter_revision=PINNED_ADAPTER_REVISION,
        expected_weight_file_sha256={str(weight): digest},
        expected_adapter_targets=["q_proj"],
        expected_adapted_layers=[19],
        expected_tokenizer_class="FakeTok",
    )
    assert rep.ok is False


def test_disable_only_adapter_rejected():
    class DisableOnly:
        def disable_adapter_layers(self):
            pass

    assert verify_adapter_off_behavior(DisableOnly())["ok"] is False


def test_adapter_restore_on_exception():
    class Flaky:
        def __init__(self):
            self.enabled = True

        def disable_adapter_layers(self):
            self.enabled = False
            raise RuntimeError("boom")

        def enable_adapter_layers(self):
            self.enabled = True

    r = verify_adapter_off_behavior(Flaky())
    assert r["restored"] is True
    assert r["ok"] is False  # disable raised


def test_manifest_must_cover_resolved_files():
    fails = validate_weight_manifest_roles(
        {"other.bin": "abc"},
        resolved_base_files=["base_model.safetensors"],
        resolved_adapter_files=["adapter_model.safetensors"],
    )
    assert len(fails) >= 2


def test_loader_policy_requires_targets_layers():
    fails = validate_loader_policy_inputs(
        base_revision=PINNED_BASE_REVISION,
        adapter_revision=PINNED_ADAPTER_REVISION,
        expected_weight_file_sha256={
            "base:/x/base_model.safetensors": "a" * 64,
            "adapter:/x/adapter_model.safetensors": "b" * 64,
        },
        expected_adapter_targets=None,
        expected_adapted_layers=None,
        torch_dtype=REQUIRED_TORCH_DTYPE,
        attn_implementation="eager",
    )
    assert any("targets" in f for f in fails)
    assert any("layers" in f for f in fails)


# --- C5 production lock ---


def test_forged_production_flags_rejected(tmp_path):
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    for name in ("torch-2.5.1-py3-none-any.whl", "numpy-2.1.3-py3-none-any.whl"):
        (wheel_dir / name).write_bytes(b"x")
    doc = generate_lock_from_wheel_dir(
        wheel_dir, repo_root=REPO, source="built_image", production_image_lock=True
    )
    d = doc.to_dict()
    # Forge flags without intended_image / full packages
    d["production_image_lock"] = True
    d["source"] = "built_image"
    d["intended_image"] = {}
    assert claim_production_lock_exists(d) is False

    d["intended_image"] = {
        "image_tag": "q1:test",
        "image_digest": "sha256:dead",
        "python_version": "3.11.15",
        "cuda_version": "12.4",
    }
    # Still missing peft/transformers packages
    assert claim_production_lock_exists(d, require_rehash=False) is False


def test_p1_p5_still_unresolved_doc():
    text = (
        REPO / "results/q1_implementation_prep/P1_P5_TRACEABILITY_MATRIX.md"
    ).read_text()
    assert "UNRESOLVED" in text
    assert "invent" in text.lower() or "unavailable" in text.lower()
