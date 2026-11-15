"""B1 production pipeline wiring — synthetic/fail-closed tests (no pretrained loads)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from traces.q1.pipeline_adapters import (
    assemble_bundles_from_condition_records,
    compute_fold_topk_selections,
    mock_condition_records_for_unit,
)
from traces.q1.production_pipeline import (
    ProductionHookConfig,
    build_production_live_hooks,
    condition_id_for,
    verify_frozen_catalog_completeness,
)
from traces.q1.scientific_execution import UnitSpec, scientific_auth_status
from traces.q1.conditions import enumerate_primary_l19_conditions

REPO = Path(__file__).resolve().parents[1]


def test_b1_primary_catalog_complete_no_silent_omission():
    report = verify_frozen_catalog_completeness(family="l19", include_downstream56=True)
    assert report.ok is True, report.failures
    assert report.n_conditions == len(
        enumerate_primary_l19_conditions(include_downstream56=True)
    )
    assert report.kind_counts.get("crossfit_topk", 0) == 20
    assert report.kind_counts.get("pair", 0) == 128  # 64 × 2 dirs


def test_b1_omitting_downstream56_fails_closed():
    report = verify_frozen_catalog_completeness(family="l19", include_downstream56=False)
    assert report.ok is False
    assert any("downstream56" in f or "omits" in f for f in report.failures)


def test_b1_secondary_catalog_complete():
    report = verify_frozen_catalog_completeness(family="l1523")
    assert report.ok is True, report.failures


def test_b1_production_hooks_refuse_without_sci_auth(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    assert scientific_auth_status().authorized is False
    hooks = build_production_live_hooks(
        ProductionHookConfig(
            repo_root=REPO,
            expected_weight_file_sha256={"base:model.safetensors": "a" * 64},
        )
    )
    spec = UnitSpec(
        model_unit="exploratory_l1523_seed42",
        family="l1523",
        role="exploratory_dev",
        seed=42,
    )
    with pytest.raises(PermissionError):
        hooks.load_model_for_unit(spec)


def test_b1_production_hooks_refuse_without_weight_manifest(monkeypatch):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    # F5 marker may still block — either way must not invent hashes / load
    hooks = build_production_live_hooks(
        ProductionHookConfig(repo_root=REPO, expected_weight_file_sha256=None)
    )
    spec = UnitSpec(
        model_unit="primary_l19_seed45",
        family="l19",
        role="primary_confirmatory",
        seed=45,
    )
    with pytest.raises((RuntimeError, PermissionError)):
        hooks.load_model_for_unit(spec)
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_b1_donors_refuse_summary_only_without_prompt_builder(monkeypatch):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    hooks = build_production_live_hooks(
        ProductionHookConfig(
            repo_root=REPO,
            expected_weight_file_sha256={"base:x": "a" * 64},
            prompt_builder=None,
            development_ids=list(range(4)),
        )
    )
    spec = UnitSpec(
        model_unit="exploratory_l1523_seed42",
        family="l1523",
        role="exploratory_dev",
        seed=42,
    )
    with pytest.raises((RuntimeError, PermissionError)):
        hooks.capture_donors_for_unit(spec, {"model": object(), "tokenizer": object()})
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_b1_effects_from_condition_records_not_summaries():
    spec = UnitSpec(
        model_unit="exploratory_l1523_seed42",
        family="l1523",
        role="exploratory_dev",
        seed=42,
    )
    ids = [str(i) for i in range(8)]
    records = mock_condition_records_for_unit(
        spec, ids, nominated_effect=2.0, intact_tt_baseline=5.0
    )
    # Patched margins are not effects: TT - patched = effect
    boundary, intervention = assemble_bundles_from_condition_records(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        records=records,
        trigger_asr=0.9,
        clean_false_fire=0.01,
        first_token_validity=0.95,
        allocation_ids=ids,
    )
    # Nominated pair (6,5) forward effect ≈ 2.0 from records
    eff = intervention.pair_effects_fwd[(6, 5)]
    assert np.allclose(eff, 2.0, atol=0.05)
    # Raw TT margins are not equal to effects
    assert not np.allclose(boundary.m_TT, eff, atol=0.5)


def test_b1_crossfit_selection_is_fold_specific():
    # Synthetic per-head per-prompt effects
    rng = np.random.default_rng(0)
    heads = list(sorted([(15, h) for h in range(8)] + [(16, h) for h in range(8)]))
    effects = {h: rng.normal(size=20) for h in heads}
    # Make head (15, 0) dominate fold A only
    effects[(15, 0)][:10] = 10.0
    effects[(16, 7)][10:] = 10.0
    sels = compute_fold_topk_selections(
        effects, fold_a_idx=list(range(10)), fold_b_idx=list(range(10, 20)), k_values=(1,)
    )
    by_fold = {s.fold_id: s for s in sels}
    assert by_fold["A"].selected_heads[0] == (15, 0)
    assert by_fold["B"].selected_heads[0] == (16, 7)
    assert by_fold["A"].joint_key != by_fold["B"].joint_key


def test_b1_condition_ids_stable():
    from traces.q1.conditions import Condition
    from traces.q1.patching import HeadRef

    c = Condition(
        kind="pair",
        direction="forward",
        heads=(HeadRef(15, 6), HeadRef(16, 5)),
    )
    assert condition_id_for(c) == "pair|forward|L15.h6+L16.h5"


def test_b1_exploratory_stopping_before_confirmatory_documented():
    """Structural: production hooks exist; live pipeline still gates confirmatory."""
    from traces.q1.scientific_execution import CONFIRMATORY_ROLES, run_live_scientific_pipeline

    assert "primary_confirmatory" in CONFIRMATORY_ROLES
    assert callable(run_live_scientific_pipeline)
    hooks = build_production_live_hooks(ProductionHookConfig(repo_root=REPO))
    assert hooks.load_model_for_unit is not None
    assert hooks.capture_donors_for_unit is not None
    assert hooks.execute_unit is not None
