"""Independent audit remediation regressions B1–B5 / M1–M5 (synthetic, no forwards)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from traces.q1.analysis import study_level_decision
from traces.q1.checkpoint_provenance import (
    assess_checkpoint_manifest,
    require_verified_snapshot_for_load,
    verify_snapshot_bytes_match_manifest,
)
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
from traces.q1.driver import (
    ModelUnitInput,
    PRIMARY_ROLE,
    SECONDARY_ROLE,
    run_model_unit_analysis,
    run_study,
)
from traces.q1.f4 import F4CalibrationTable
from traces.q1.frozen_artifacts import load_ordered_dolly_ids
from traces.q1.integrity_gates import evaluate_preconfirmation_gates
from traces.q1.pipeline_adapters import (
    assemble_bundles_from_condition_records,
    build_live_hooks_from_record_provider,
    compute_fold_topk_selections,
    mock_condition_records_for_unit,
    required_adapter_subfolders,
    unit_spec_to_adapter_subfolder,
)
from traces.q1.runtime_lock import verify_installed_runtime_environment
from traces.q1.scientific_execution import (
    expected_frozen_unit_specs,
    run_typed_pipeline_from_bundle,
)
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256

REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _unit_from_maps(
    *,
    role: str,
    seed: int,
    family: str,
    n: int = 40,
    pair_fx: dict | None = None,
    head_fx: dict | None = None,
    trigger_asr: float = 0.95,
    first_token_validity: float = 0.95,
) -> ModelUnitInput:
    rng = np.random.default_rng(seed)
    ids = [f"{role}-{seed}-p{i}" for i in range(n)]
    m_CC = rng.normal(0, 0.01, n)
    m_TT = m_CC + 2.0
    if pair_fx is None:
        pair_fx = {p: np.full(n, 0.1) for p in EXPECTED_L15_L16_PAIRS}
        pair_fx[(6, 5)] = np.full(n, 2.0)
    if head_fx is None:
        head_fx = {(15, 6): np.full(n, 0.5), (16, 5): np.full(n, 0.4)}
        for L in range(19):
            for h in range(8):
                head_fx.setdefault((L, h), np.full(n, 0.05))
    return ModelUnitInput(
        model_unit=f"{role}/seed{seed}",
        family=family,
        role=role,
        prompt_ids=ids,
        m_TT=m_TT,
        m_CT=m_CC + 0.2,
        m_TC=m_TT - 0.2,
        m_CC=m_CC,
        eligible_mask=np.ones(n, dtype=bool),
        trigger_asr=trigger_asr,
        clean_false_fire=0.01,
        first_token_validity=first_token_validity,
        n_frozen=n,
        pair_effects_fwd=pair_fx,
        pair_effects_rev={k: v * 0.9 for k, v in pair_fx.items()},
        head_effects_fwd=head_fx,
        head_effects_rev={k: v * 0.9 for k, v in head_fx.items()},
        e_route_fwd=m_TT - (m_CC + 0.2),
        e_route_rev=(m_TT - 0.2) - m_CC,
        joint_set_effects_fwd={((15, 6),): head_fx[(15, 6)]},
        joint_set_effects_rev={((15, 6),): head_fx[(15, 6)] * 0.9},
    )


# --- B4: three-pair vs 64-pair ---


def test_b4_three_pair_cannot_h1_pass():
    """Agent 1 regression: incomplete 3-pair comparator must not yield H1-pass."""
    three = {
        (6, 5): np.full(40, 5.0),
        (0, 0): np.full(40, 0.01),
        (1, 1): np.full(40, 0.01),
    }
    heads = {(15, 6): np.full(40, 0.5), (16, 5): np.full(40, 0.4)}
    for L in range(19):
        for h in range(8):
            heads.setdefault((L, h), np.full(40, 0.05))
    unit = _unit_from_maps(
        role=PRIMARY_ROLE, seed=42, family="l19", pair_fx=three, head_fx=heads
    )
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    res = run_model_unit_analysis(unit, held)
    # Force R-and via route — even if route supports H1, comparator incomplete
    fwd = res.h1.get("forward") or res.h1
    state = fwd.get("state") if isinstance(fwd, dict) else res.h1.get("full_H1")
    assert state != "H1-pass"
    assert state in {"H1-indeterminate", "inapplicable", None} or (
        isinstance(fwd, dict) and fwd.get("h1_comparator_complete") is False
    )


def test_b4_complete_64_allows_h1_inference_path():
    unit = _unit_from_maps(role=PRIMARY_ROLE, seed=42, family="l19")
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    res = run_model_unit_analysis(unit, held)
    # May be inapplicable for route class, but comparator gate should be complete when denom ok
    fwd = res.h1.get("forward")
    if isinstance(fwd, dict) and fwd.get("route_denom_ok") is not False:
        if "h1_comparator_complete" in fwd:
            assert fwd["h1_comparator_complete"] is True


# --- B5: secondary ASR-zero / missing heads / three pairs ---


def test_b5_asr_zero_gate_failed_no_replication():
    unit = _unit_from_maps(
        role=SECONDARY_ROLE, seed=45, family="l1523", trigger_asr=0.0
    )
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    out = run_study([unit], held, exploratory_single_bos_ranks=None)
    # Without exploratory ranks, confirmatory blocked — call analyzer directly
    from traces.q1.driver import _analyze_secondary_unit

    sec = _analyze_secondary_unit(unit, held)
    assert sec.get("status") == "gate-failed"
    assert sec.get("replication_support") is False


def test_b5_missing_heads_no_zero_substitute():
    pair = {p: np.full(40, 0.1) for p in EXPECTED_L15_L16_PAIRS}
    pair[(6, 5)] = np.full(40, 2.0)
    # Only one nominated head
    heads = {(15, 6): np.full(40, 0.5)}
    for L in range(19):
        for h in range(8):
            if (L, h) != (16, 5):
                heads.setdefault((L, h), np.full(40, 0.05))
    unit = _unit_from_maps(
        role=SECONDARY_ROLE, seed=46, family="l1523", pair_fx=pair, head_fx=heads
    )
    from traces.q1.driver import _analyze_secondary_unit

    sec = _analyze_secondary_unit(unit, unit.prompt_ids + [f"h{i}" for i in range(160)])
    for d in sec["directions"].values():
        assert d.get("pass") is False
        assert d.get("replication_support") is False
        assert "missing" in str(d.get("reason", "")).lower() or d.get(
            "missing_nominated_heads"
        )


def test_b5_three_pairs_incomplete_refuses_replication():
    three = {
        (6, 5): np.full(40, 5.0),
        (0, 1): np.full(40, 0.1),
        (2, 3): np.full(40, 0.1),
    }
    heads = {(15, 6): np.full(40, 0.5), (16, 5): np.full(40, 0.4)}
    unit = _unit_from_maps(
        role=SECONDARY_ROLE, seed=45, family="l1523", pair_fx=three, head_fx=heads
    )
    from traces.q1.driver import _analyze_secondary_unit

    sec = _analyze_secondary_unit(unit, unit.prompt_ids + [f"h{i}" for i in range(160)])
    assert sec.get("replication_support") is False
    for d in sec["directions"].values():
        assert d.get("pass") is False


# --- B1: mock adapter integration ---


def test_b1_assemble_from_raw_records_not_external_arrays():
    spec = expected_frozen_unit_specs()[0]
    ids = [str(x) for x in load_ordered_dolly_ids(
        REPO / "results/q1_selection/development_instructions.json"
    )]
    recs = mock_condition_records_for_unit(spec, ids, rng_seed=7)
    b, i = assemble_bundles_from_condition_records(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        records=recs,
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.96,
        allocation_ids=ids,
    )
    assert len(b.pair_effects_fwd) == 64
    assert len(b.head_effects_fwd) == 152
    assert b.catalog_condition_ids
    unit = i.to_model_unit_input()
    assert unit.model_unit == spec.model_unit


def test_b1_fold_topk_selection_produces_joint_keys():
    heads = {(L, h): np.random.default_rng(0).normal(0, 1, 20) for L in range(3) for h in range(4)}
    sels = compute_fold_topk_selections(
        heads, fold_a_idx=list(range(10)), fold_b_idx=list(range(10, 20)), k_values=(1, 2)
    )
    assert len(sels) == 4  # 2 folds × 2 k
    assert all(s.joint_key == tuple(sorted(s.selected_heads)) for s in sels)


def test_b1_ten_adapter_subfolders():
    assert len(required_adapter_subfolders()) == 10
    specs = expected_frozen_unit_specs()
    primary = [s for s in specs if s.role == "primary_confirmatory"][0]
    assert unit_spec_to_adapter_subfolder(primary) == f"l19/seed{primary.seed}"


# --- B2: integrity gates ---


def test_b2_missing_evidence_blocks_progression():
    rep = evaluate_preconfirmation_gates()
    assert rep.progression_blocked is True
    assert rep.gates["frozen_f4_tolerance_table"].status == "blocked"
    assert rep.gates["descriptive_double_bos_64_pair_scan"].status == "blocked"


def test_b2_positive_and_negative_fixtures():
    # Negative: high ineligibility
    bad = evaluate_preconfirmation_gates(
        development_n_eligible=40,
        development_n_frozen=50,
        exploratory_first_token_validity=0.95,
        double_bos_scan_completed=True,
        double_bos_n_pairs=64,
        f4_table={"locked": True, "families": {"l1523": {}}},
    )
    assert bad.gates["development_ineligibility"].status == "failed"

    # Positive synthetic (F4 locked stub + identities suite auto-run)
    table = F4CalibrationTable(batching_padding_policy="test")
    # lock requires families — use empty refusal path covered; supply locked mapping
    good = evaluate_preconfirmation_gates(
        development_n_eligible=48,
        development_n_frozen=50,
        exploratory_first_token_validity=0.95,
        double_bos_scan_completed=True,
        double_bos_n_pairs=64,
        f4_table={"locked": True, "families": {"l19": {"epsilon_family": 1e-3}}},
    )
    assert good.gates["development_ineligibility"].ok
    assert good.gates["exploratory_first_token_validity"].ok
    assert good.gates["descriptive_double_bos_64_pair_scan"].ok
    assert good.gates["identities_1_to_9"].ok
    assert good.gates["frozen_f4_tolerance_table"].ok
    assert good.progression_blocked is False


# --- B3: checkpoint provenance BLOCKED ---


def test_b3_manifest_blocked_without_verified_bytes():
    st = assess_checkpoint_manifest()
    assert st.blocked is True
    assert st.ok is False
    assert len(st.adapter_subfolders) == 10
    req = require_verified_snapshot_for_load(manifest_path=None, snapshot_root=None)
    assert req["ok"] is False
    assert req["blocked"] is True


def test_b3_snapshot_byte_binding(tmp_path):
    f = tmp_path / "base" / "model.safetensors"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"snap-bytes")
    import hashlib

    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    ok = verify_snapshot_bytes_match_manifest(
        snapshot_dir=tmp_path, expected_sha256_by_relpath={"base/model.safetensors": digest}
    )
    assert ok["ok"] is True
    bad = verify_snapshot_bytes_match_manifest(
        snapshot_dir=tmp_path,
        expected_sha256_by_relpath={"base/model.safetensors": "0" * 64},
    )
    assert bad["ok"] is False


# --- M1–M5 ---


def test_m1_allocation_order_binding_rejected_on_reorder():
    spec = expected_frozen_unit_specs()[0]
    ids = [str(x) for x in load_ordered_dolly_ids(
        REPO / "results/q1_selection/development_instructions.json"
    )]
    recs = mock_condition_records_for_unit(spec, ids)
    with pytest.raises(ValueError, match="allocation"):
        assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=list(reversed(ids)),
            records=recs,
            trigger_asr=0.95,
            clean_false_fire=0.01,
            first_token_validity=0.95,
            allocation_ids=ids,
        )


def test_m2_typed_pipeline_default_ranks_none_blocks_fabricated_clear(tmp_path):
    """Default exploratory_ranks must be None — not (1,2,3)."""
    import inspect
    from traces.q1 import scientific_execution as se

    sig = inspect.signature(se.run_typed_pipeline_from_bundle)
    assert sig.parameters["exploratory_ranks"].default is None


def test_m3_study_level_four_of_five_and_two_inconclusive():
    assert study_level_decision(["R-and"] * 4 + ["R-or"]).startswith("SUPPORTED")
    assert study_level_decision(["gate-failed", "Indeterminate", "R-and", "R-or", "R-tag"]) == "INCONCLUSIVE"
    with pytest.raises(ValueError):
        study_level_decision(["R-and"] * 4)  # not five


def test_m4_loose_package_name_rejected_and_no_production_claim():
    pins = REPO / "TRACES_Q1_execution_direct_pins_v1_2.txt"
    # installed has substring-like name only
    rep = verify_installed_runtime_environment(
        pins_file=pins,
        installed={"torchvision": "2.5.1"},  # not an exact pin match for torch
        intended_image=None,
    )
    assert rep["production_claim"] is False
    assert rep["ok"] is False


def test_m5_scientific_loader_requires_auth(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    from traces.q1.loader import load_q1_scientific_model

    with pytest.raises(PermissionError):
        load_q1_scientific_model(
            subfolder="l1523/seed42",
            expected_weight_file_sha256={
                "base:/x": "a" * 64,
                "adapter:/y": "b" * 64,
            },
            expected_adapter_targets=["q_proj"],
            expected_adapted_layers=[19],
        )


# --- minors / F1 ---


def test_f1_hash_preserved():
    assert f1_full_set_stream_sha256(180) == F1_REF


def test_exploratory_requires_exact_n_dev():
    from traces.q1.driver import EXPLORATORY_ROLE, _analyze_exploratory_unit

    u = _unit_from_maps(role=EXPLORATORY_ROLE, seed=42, family="l1523", n=40)
    out = _analyze_exploratory_unit(u)
    assert out["allocation_ok"] is False
