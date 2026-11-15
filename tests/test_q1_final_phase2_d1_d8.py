"""Final Phase 2 corrections D1–D8 — negative regression tests (no forwards)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from traces.q1.analysis import Interval, evaluate_h1_direction
from traces.q1.driver import (
    ADAPTER_OFF_ROLE,
    EXPLORATORY_ROLE,
    PRIMARY_ROLE,
    SECONDARY_ROLE,
    ModelUnitInput,
    run_model_unit_analysis,
    run_study,
)
from traces.q1.frozen_artifacts import FROZEN_FILE_SHA256, verify_frozen_artifacts
from traces.q1.loader import validate_loader_policy_inputs
from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
from traces.q1.post_load import validate_weight_manifest_roles
from traces.q1.runtime_lock import claim_production_lock_exists
from traces.q1.runtime_spec import REQUIRED_TORCH_DTYPE
from traces.q1.scientific_execution import (
    F5_APPROVED_STATUS_EXACT,
    ConditionOutputBundle,
    LiveForwardHooks,
    UnitSpec,
    expected_frozen_unit_specs,
    f5_approval_state,
    reconcile_boundary_intervention,
    run_live_scientific_pipeline,
    run_scientific_preflight,
)
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256

REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def test_f1_hash_preserved():
    assert f1_full_set_stream_sha256(180) == F1_REF


def _bundle(n=10, n_elig=8, bad: str | None = None) -> ConditionOutputBundle:
    ids = [f"p{i}" for i in range(n)]
    mask = np.zeros(n, dtype=bool)
    mask[:n_elig] = True
    if bad == "reorder_elig_ids":
        elig_ids = list(reversed([ids[i] for i in range(n_elig)]))
    else:
        elig_ids = [ids[i] for i in range(n_elig)]
    m = np.ones(n_elig if bad != "full_len" else n, dtype=np.float64)
    if bad == "nonfinite":
        m = m.copy()
        m[0] = np.nan
    heads = {(0, 0): np.ones(n_elig)}
    pairs = {(0, 0): np.ones(n_elig)}
    return ConditionOutputBundle(
        model_unit="primary_l19_seed42",
        family="l19",
        role=PRIMARY_ROLE,
        prompt_ids=ids if bad != "dup_ids" else ids + ["p0"],
        m_TT=m,
        m_CT=m,
        m_TC=m,
        m_CC=m,
        eligible_mask=mask if bad != "mask_len" else np.ones(n + 1, dtype=bool),
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.95,
        n_frozen=n,
        pair_effects_fwd=pairs,
        pair_effects_rev=pairs,
        head_effects_fwd=heads,
        head_effects_rev=heads,
        e_route_fwd=m,
        e_route_rev=m,
        joint_set_effects_fwd={((0, 0),): np.ones(n_elig)},
        joint_set_effects_rev={((0, 0),): np.ones(n_elig)},
        eligible_prompt_ids=elig_ids,
    )


# --- D1 ---


def test_d1_rejects_full_denominator_where_eligible_required():
    b = _bundle(bad="full_len")
    with pytest.raises(ValueError, match="full-denominator|eligible-only"):
        b.to_model_unit_input()


def test_d1_rejects_mask_length_and_ordering():
    with pytest.raises(ValueError):
        _bundle(bad="mask_len").to_model_unit_input()
    with pytest.raises(ValueError, match="ordering"):
        _bundle(bad="reorder_elig_ids").to_model_unit_input()


def test_d1_rejects_nonfinite():
    with pytest.raises(ValueError, match="nonfinite"):
        _bundle(bad="nonfinite").to_model_unit_input()


def test_d1_accepts_consistent_eligible_only():
    unit = _bundle().to_model_unit_input()
    assert int(unit.eligible_mask.sum()) == 8
    assert len(unit.m_TT) == 8


# --- D2 ---


def test_d2_frozen_artifacts_match_baselines():
    rep = verify_frozen_artifacts(REPO)
    assert rep["ok"] is True
    for rel, expected in FROZEN_FILE_SHA256.items():
        assert rep["checked"][rel]["match"] is True
        assert rep["checked"][rel]["sha256"] == expected


def test_d2_preflight_compares_hashes_not_just_computes():
    pre = run_scientific_preflight(REPO, require_sci_auth=False)
    assert "frozen_artifacts" in pre.checks
    # Still fails overall (F5/lock/auth paths) but frozen check should be present
    assert pre.checks["frozen_artifacts"]["ok"] is True


def test_d2_rejects_reordered_ids(tmp_path):
    # Copy held-out and reorder
    src = REPO / "results/q1_selection/held_out_instructions.json"
    data = json.loads(src.read_text())
    data = list(reversed(data))
    # Build a fake root with only this file wrong — verify_frozen_artifacts on REPO
    # already passes; unit-test sequence helper
    from traces.q1.frozen_artifacts import (
        FROZEN_HELD_OUT_ID_SEQUENCE_SHA256,
        ordered_id_sequence_sha256,
    )

    ids = [r["dolly_id"] for r in data]
    assert ordered_id_sequence_sha256(ids) != FROZEN_HELD_OUT_ID_SEQUENCE_SHA256


# --- D3 ---


def test_d3_production_lock_requires_rehash_and_all_pins(tmp_path):
    forged = {
        "schema_version": "1.0",
        "protocol_version": "1.3",
        "frozen_scientific_commit": "b1e8874f45311678e6fa241f504e7e09d4a7808c",
        "production_image_lock": True,
        "source": "built_image",
        "intended_image": {
            "image_tag": "q1:x",
            "image_digest": "sha256:abc",
            "python_version": "3.11.15",
            "cuda_version": "12.4",
        },
        "direct_pins": {
            "match": True,
            "sha256": "4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048",
            "expected_sha256": "4a90dd7133719a751400c456db615e726a245d9a34073b7fb64243bf2a212048",
        },
        "wheels": [
            {
                "name": "torch",
                "version": "2.5.1",
                "filename": "torch-2.5.1-py3-none-any.whl",
                "sha256": "a" * 64,
            }
        ],
    }
    # No wheel_dir / require_rehash False → reject
    assert claim_production_lock_exists(forged, require_rehash=False) is False
    assert claim_production_lock_exists(forged, wheel_dir=None) is False
    # Synthetic source rejected
    forged["source"] = "synthetic_fixture"
    assert (
        claim_production_lock_exists(
            forged, wheel_dir=tmp_path, require_rehash=True
        )
        is False
    )


def test_d3_preflight_production_lock_blocked():
    pre = run_scientific_preflight(REPO, require_sci_auth=False)
    assert any("production runtime lock" in f for f in pre.failures)


# --- D4 ---


def _unit(role, seed, family, n=20) -> ModelUnitInput:
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
        model_unit=f"{role}_seed{seed}" if role != ADAPTER_OFF_ROLE else "adapter_off_base_reference",
        family=family,
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


def test_d4_absent_exploratory_blocks_confirmatory():
    units = [_unit(PRIMARY_ROLE, 42, "l19")]
    held = [f"h{i}" for i in range(200)]
    out = run_study(units, held, exploratory_single_bos_ranks=None)
    assert out["exploratory_stop"]["confirmatory_blocked"] is True
    assert any(p.get("skipped") for p in out["per_unit"])


def test_d4_partial_seeds_or_ranks_block():
    units = [_unit(EXPLORATORY_ROLE, 42, "l1523"), _unit(PRIMARY_ROLE, 42, "l19")]
    held = [f"h{i}" for i in range(200)]
    out = run_study(units, held, exploratory_single_bos_ranks=[1, 2, 3])
    assert out["exploratory_stop"]["confirmatory_blocked"] is True


def test_d4_stop_on_rank_criterion():
    units = [_unit(EXPLORATORY_ROLE, s, "l1523") for s in (42, 43, 44)]
    units += [_unit(PRIMARY_ROLE, 42, "l19")]
    held = [f"h{i}" for i in range(200)]
    out = run_study(units, held, exploratory_single_bos_ranks=[11, 12, 1])
    assert out["exploratory_stop"]["triggered"] is True
    assert out["exploratory_stop"]["confirmatory_blocked"] is True


def test_d4_no_default_fabricated_ranks():
    units = [_unit(EXPLORATORY_ROLE, s, "l1523") for s in (42, 43, 44)]
    held = [f"h{i}" for i in range(200)]
    out = run_study(units, held)  # ranks absent
    assert out["exploratory_stop"].get("fabricated_ranks") is False
    assert out["exploratory_stop"]["confirmatory_blocked"] is True


# --- D5 ---


def test_d5_h1_direction_by_route():
    rank = Interval(1, 2, 1)
    share = Interval(0.6, 0.8, 0.7)
    marg = Interval(0.2, 0.4, 0.3)
    # R-and: forward applicable, reverse inapplicable
    fwd = evaluate_h1_direction(
        route_class="R-and",
        direction="forward",
        rank_iv=rank,
        share_iv=share,
        marginal_h15_iv=marg,
        marginal_h16_iv=marg,
        route_denom_ok=True,
    )
    rev = evaluate_h1_direction(
        route_class="R-and",
        direction="reverse",
        rank_iv=rank,
        share_iv=share,
        marginal_h15_iv=marg,
        marginal_h16_iv=marg,
        route_denom_ok=True,
    )
    assert fwd["state"] == "H1-pass"
    assert rev["state"] == "inapplicable"
    # R-or: reverse applicable
    rev_or = evaluate_h1_direction(
        route_class="R-or",
        direction="reverse",
        rank_iv=rank,
        share_iv=share,
        marginal_h15_iv=marg,
        marginal_h16_iv=marg,
        route_denom_ok=True,
    )
    fwd_or = evaluate_h1_direction(
        route_class="R-or",
        direction="forward",
        rank_iv=rank,
        share_iv=share,
        marginal_h15_iv=marg,
        marginal_h16_iv=marg,
        route_denom_ok=True,
    )
    assert rev_or["state"] == "H1-pass"
    assert fwd_or["state"] == "inapplicable"


def test_d5_driver_selects_reverse_for_r_or(monkeypatch):
    # Build unit and force route class via monkeypatch of run_route_analysis
    from traces.q1 import driver as drv

    unit = _unit(PRIMARY_ROLE, 1, "l19", n=40)
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]

    real = drv.run_route_analysis

    def fake_route(contrasts, draws):
        out = real(contrasts, draws)
        out["route_class"] = "R-or"
        return out

    monkeypatch.setattr(drv, "run_route_analysis", fake_route)
    res = run_model_unit_analysis(unit, held)
    assert res.h1.get("applicable_direction") == "reverse"
    assert res.h1.get("full_H1") == res.h1["reverse"].get("state")


def test_d5_driver_selects_forward_for_r_and(monkeypatch):
    from traces.q1 import driver as drv

    unit = _unit(PRIMARY_ROLE, 2, "l19", n=40)
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]

    real = drv.run_route_analysis

    def fake_route(contrasts, draws):
        out = real(contrasts, draws)
        out["route_class"] = "R-and"
        return out

    monkeypatch.setattr(drv, "run_route_analysis", fake_route)
    res = run_model_unit_analysis(unit, held)
    assert res.h1.get("applicable_direction") == "forward"
    assert res.h1.get("full_H1") == res.h1["forward"].get("state")


# --- D6 ---


def test_d6_rejects_substring_base_adapter_names():
    fails = validate_weight_manifest_roles(
        {"/tmp/mybase_weights.bin": "a" * 64, "/tmp/notreallyadapter.bin": "b" * 64}
    )
    assert any("base:<path>" in f for f in fails)
    assert any("adapter:<path>" in f for f in fails)


def test_d6_requires_exact_resolved_coverage(tmp_path):
    base = tmp_path / "model.safetensors"
    adapter = tmp_path / "adapter_model.safetensors"
    base.write_bytes(b"b")
    adapter.write_bytes(b"a")
    fails = validate_weight_manifest_roles(
        {f"base:{base}": "a" * 64, f"adapter:{adapter}": "b" * 64},
        resolved_base_files=[str(base)],
        resolved_adapter_files=[str(adapter) + ".missing"],
        require_resolved_inventories=True,
    )
    assert any("missing resolved adapter" in f for f in fails)


def test_d6_loader_policy_role_tags():
    fails = validate_loader_policy_inputs(
        base_revision=PINNED_BASE_REVISION,
        adapter_revision=PINNED_ADAPTER_REVISION,
        expected_weight_file_sha256={
            "something_with_base_in_name.safetensors": "a" * 64,
            "something_adapterish.safetensors": "b" * 64,
        },
        expected_adapter_targets=["q_proj"],
        expected_adapted_layers=[19],
        torch_dtype=REQUIRED_TORCH_DTYPE,
        attn_implementation="eager",
    )
    assert any("base:" in f or "adapter:" in f for f in fails)


# --- D7 ---


def test_d7_expected_unit_set_complete():
    specs = expected_frozen_unit_specs()
    assert len(specs) == 3 + 5 + 2 + 1
    assert [s.seed for s in specs if s.role == EXPLORATORY_ROLE] == [42, 43, 44]


def test_d7_reconcile_rejects_mismatch():
    b1 = _bundle()
    b2 = _bundle()
    b2.model_unit = "other"
    with pytest.raises(ValueError, match="mismatch"):
        reconcile_boundary_intervention(b1, b2)


def test_d7_live_pipeline_requires_new_hooks(monkeypatch):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    # Preflight still fails (F5/lock) before hooks — PermissionError or preflight
    with pytest.raises((PermissionError, RuntimeError)):
        run_live_scientific_pipeline(
            repo_root=REPO,
            held_out_ids_allocation_order=[f"p{i}" for i in range(200)],
            hooks=LiveForwardHooks(),  # missing callables
            out_dir=REPO / "results/q1_implementation_prep/_tmp_d7",
        )


# --- D8 ---


def test_d8_not_approved_substring_does_not_authorize(tmp_path, monkeypatch):
    marker = tmp_path / "F5_OWNER_APPROVAL.marker"
    marker.write_text("F5_STATUS=NOT APPROVED\nOWNER_SIGNER=owner\n")
    # Point f5 at tmp by using repo_root structure
    prep = tmp_path / "results/q1_implementation_prep"
    prep.mkdir(parents=True)
    (prep / "F5_OWNER_APPROVAL.marker").write_text(
        "F5_STATUS=NOT APPROVED\nOWNER_SIGNER=owner\n"
    )
    monkeypatch.setenv("TRACES_Q1_F5_OWNER_APPROVED", "1")
    st = f5_approval_state(repo_root=tmp_path)
    assert st["approved"] is False
    assert st["pending"] is True


def test_d8_exact_status_required(tmp_path, monkeypatch):
    prep = tmp_path / "results/q1_implementation_prep"
    prep.mkdir(parents=True)
    (prep / "F5_OWNER_APPROVAL.marker").write_text(
        f"{F5_APPROVED_STATUS_EXACT}\nOWNER_SIGNER=research_owner\n"
    )
    monkeypatch.setenv("TRACES_Q1_F5_OWNER_APPROVED", "1")
    st = f5_approval_state(repo_root=tmp_path)
    assert st["approved"] is True
    # Marker existence alone insufficient
    monkeypatch.delenv("TRACES_Q1_F5_OWNER_APPROVED", raising=False)
    st2 = f5_approval_state(repo_root=tmp_path)
    assert st2["approved"] is False


def test_d8_current_repo_f5_still_pending():
    st = f5_approval_state(repo_root=REPO)
    assert st["pending"] is True
    assert st["approved"] is False


def test_p1_p5_unresolved_retained():
    text = (REPO / "results/q1_implementation_prep/P1_P5_TRACEABILITY_MATRIX.md").read_text()
    assert "UNRESOLVED" in text
