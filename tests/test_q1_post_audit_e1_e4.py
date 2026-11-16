"""Post-audit E1–E4 corrections (synthetic CPU; no pretrained forwards)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
from traces.q1.analysis import pair_effect_forward, pair_effect_reverse
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
from traces.q1.driver import SECONDARY_ROLE, ModelUnitInput, _analyze_secondary_unit
from traces.q1.frozen_artifacts import load_ordered_dolly_ids
from traces.q1.generation import run_f3_generation
from traces.q1.integrity_gates import evaluate_preconfirmation_gates
from traces.q1.pipeline_adapters import (
    PromptConditionRecord,
    assemble_bundles_from_condition_records,
    mock_condition_records_for_unit,
)
from traces.q1.scientific_execution import (
    LiveForwardHooks,
    UnitSpec,
    expected_frozen_unit_specs,
    run_live_scientific_pipeline,
    stash_mock_observations_on_donors,
)
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256

REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _frozen_dev():
    return load_ordered_dolly_ids(
        REPO / "results/q1_selection/development_instructions.json"
    )


def _frozen_held_out():
    return load_ordered_dolly_ids(
        REPO / "results/q1_selection/held_out_instructions.json"
    )


def _empirical_integrity_evidence(tmp_path=None) -> dict[str, Any]:
    """Persisted fabricated artifacts with independently checkable hashes."""
    from traces.q1.empirical_evidence import (
        write_fabricated_double_bos_artifact,
        write_fabricated_f4_artifact,
        write_fabricated_identity_artifact,
    )

    root = Path(tmp_path) if tmp_path is not None else Path("/tmp/traces_q1_e2_fixtures")
    root.mkdir(parents=True, exist_ok=True)
    id_p, id_s = write_fabricated_identity_artifact(root / "identity.json")
    f4_p, f4_s = write_fabricated_f4_artifact(root / "f4.json")
    db_p, db_s = write_fabricated_double_bos_artifact(root / "double_bos.json")
    return {
        "identity_artifact": {"artifact_path": str(id_p), "sha256": id_s},
        "f4_artifact": {"artifact_path": str(f4_p), "sha256": f4_s},
        "double_bos_artifact": {"artifact_path": str(db_p), "sha256": db_s},
    }


def _pass_preflight(monkeypatch):
    from traces.q1.scientific_execution import PreflightReport

    def _ok(*_a, **_k):
        return PreflightReport(ok=True, failures=[], checks={"synthetic": True})

    monkeypatch.setattr(
        "traces.q1.scientific_execution.run_scientific_preflight", _ok
    )
    monkeypatch.setattr(
        "traces.q1.scientific_execution.require_scientific_authorization",
        lambda: None,
    )


def _bundle_for(spec: UnitSpec, *, nominated_effect: float = 2.0):
    from traces.q1.pipeline_adapters import assemble_bundles_from_condition_records

    ids = (
        [str(x) for x in _frozen_dev()]
        if spec.role == "exploratory_dev"
        else [f"p{i}" for i in range(8)]
    )
    recs = mock_condition_records_for_unit(
        spec,
        ids,
        nominated_effect=nominated_effect,
        intact_tt_baseline=5.0,
        intact_cc_baseline=1.0,
    )
    b, i = assemble_bundles_from_condition_records(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        records=recs,
        trigger_asr=0.95,
        clean_false_fire=0.01,
        first_token_validity=0.96,
        allocation_ids=ids if spec.role == "exploratory_dev" else None,
    )
    return i.to_model_unit_input() if hasattr(i, "to_model_unit_input") else b


def _spy_hooks(
    *,
    nominated_effect_by_seed: dict[int, float] | None = None,
) -> tuple[LiveForwardHooks, dict[str, list[str]]]:
    calls: dict[str, list[str]] = {"load": [], "donors": [], "execute": []}
    effects = nominated_effect_by_seed or {}

    def load(spec: UnitSpec) -> dict[str, Any]:
        calls["load"].append(spec.model_unit)
        return {
            "model": object(),
            "tokenizer": object(),
            "model_unit": spec.model_unit,
            "family": spec.family,
            "role": spec.role,
            "seed": spec.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION
            if spec.family != "base"
            else None,
            "weight_file_sha256": {
                f"base:/fake/{spec.model_unit}.safetensors": "a" * 64,
                f"adapter:/fake/{spec.model_unit}_adapter.safetensors": "b" * 64,
            },
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }

    def donors(spec: UnitSpec, _model: Any) -> dict[str, Any]:
        calls["donors"].append(spec.model_unit)
        ids = (
            [str(x) for x in _frozen_dev()]
            if spec.role == "exploratory_dev"
            else [f"p{i}" for i in range(8)]
        )
        return {
            "n_donor_caches": len(ids),
            "clean_references_frozen": True,
            "instruction_ids": ids,
        }

    def execute(spec: UnitSpec, _model: Any, donors_ev: Any) -> tuple:
        calls["execute"].append(spec.model_unit)
        nom = 2.0
        if spec.seed is not None and spec.seed in effects:
            nom = effects[spec.seed]
        unit = _bundle_for(spec, nominated_effect=nom)
        # Return boundary+intervention bundles via unit's source; reuse unit twice
        # Live path expects ConditionOutputBundle pair — use adapter assemble
        ids = (
            [str(x) for x in _frozen_dev()]
            if spec.role == "exploratory_dev"
            else [f"p{i}" for i in range(8)]
        )
        recs = mock_condition_records_for_unit(spec, ids, nominated_effect=nom)
        b, i = assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.95,
            clean_false_fire=0.01,
            first_token_validity=0.96,
            allocation_ids=ids if spec.role == "exploratory_dev" else None,
        )
        stash_mock_observations_on_donors(
            donors_ev, spec=spec, records=recs, prompt_ids=ids
        )
        return b, i

    hooks = LiveForwardHooks(
        load_model_for_unit=load,
        capture_donors_for_unit=donors,
        execute_unit=execute,
    )
    return hooks, calls


def _confirmatory_units() -> set[str]:
    return {
        s.model_unit
        for s in expected_frozen_unit_specs()
        if s.role in {"primary_confirmatory", "secondary_replication"}
    }


# --- E1: true intervention effects ---


def test_e1_effects_are_signed_contrasts_not_raw_margins():
    spec = expected_frozen_unit_specs()[0]
    ids = [str(x) for x in _frozen_dev()]
    tt_base, cc_base = 7.5, -2.0
    nominated = 3.0
    recs = mock_condition_records_for_unit(
        spec,
        ids,
        nominated_effect=nominated,
        other_pair_effect=0.25,
        intact_tt_baseline=tt_base,
        intact_cc_baseline=cc_base,
        rng_seed=11,
    )
    # Raw patched margins for nominated forward must equal TT - effect ≠ effect
    fwd_cid = "pair|forward|L15.h6+L16.h5"
    patched_margins = [r.margin for r in recs if r.condition_id == fwd_cid]
    assert patched_margins
    assert abs(np.mean(patched_margins) - nominated) > 1.0  # raw ≠ effect

    _b, i = assemble_bundles_from_condition_records(
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
    eff = i.pair_effects_fwd[(6, 5)]
    assert np.allclose(eff, nominated, atol=0.05)
    # Reverse: effect ≈ nominated (patched = CC + effect)
    eff_r = i.pair_effects_rev[(6, 5)]
    assert np.allclose(eff_r, nominated, atol=0.05)
    # Explicit contrast identity
    tt = i.m_TT
    # Reconstruct patched from contrast: patched = TT - effect
    patched = tt - eff
    assert np.allclose(pair_effect_forward(tt, patched), eff)


def test_e1_rejects_missing_intact_and_duplicates():
    spec = expected_frozen_unit_specs()[0]
    ids = ["a", "b", "c"]
    recs = mock_condition_records_for_unit(spec, ids, include_upstream_heads=False)
    # Drop intact TT
    recs = [r for r in recs if r.kind != "intact_TT"]
    with pytest.raises(KeyError, match="TT"):
        assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=recs,
            trigger_asr=0.95,
            clean_false_fire=0.01,
            first_token_validity=0.95,
        )
    # Duplicates
    full = mock_condition_records_for_unit(spec, ids, include_upstream_heads=False)
    with pytest.raises(ValueError, match="duplicate"):
        assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=full + full[:1],
            trigger_asr=0.95,
            clean_false_fire=0.01,
            first_token_validity=0.95,
        )


def test_e1_rejects_wrong_direction_and_invalid_heads():
    spec = expected_frozen_unit_specs()[0]
    ids = ["a", "b"]
    bad = PromptConditionRecord(
        instruction_id="a",
        condition_id="pair|sideways|L15.h6+L16.h5",
        direction="sideways",
        margin=1.0,
        eligible=True,
        donor_instruction_id="a",
        clean_reference_token=0,
        kind="pair",
        heads=((15, 6), (16, 5)),
        model_unit=spec.model_unit,
    )
    base = mock_condition_records_for_unit(spec, ids, include_upstream_heads=False)
    with pytest.raises(ValueError, match="direction"):
        assemble_bundles_from_condition_records(
            model_unit=spec.model_unit,
            family=spec.family,
            role=spec.role,
            prompt_ids=ids,
            records=base + [bad],
            trigger_asr=0.95,
            clean_false_fire=0.01,
            first_token_validity=0.95,
        )


# --- E2: integrity gates wired; spy zero confirmatory ---


@pytest.mark.parametrize(
    "drop_key",
    ["identity_artifact", "f4_artifact", "double_bos_artifact"],
)
def test_e2_spy_zero_confirmatory_per_failed_integrity_prerequisite(
    tmp_path, monkeypatch, drop_key
):
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks(
        nominated_effect_by_seed={42: 2.0, 43: 2.0, 44: 2.0}
    )
    ev = _empirical_integrity_evidence(tmp_path)
    del ev[drop_key]
    # Adversarial stubs that must NOT clear the gate
    if drop_key == "identity_artifact":
        from traces.q1.identities import run_synthetic_identity_suite

        ev["identity_suite"] = run_synthetic_identity_suite()
    elif drop_key == "f4_artifact":
        ev["f4_table"] = {
            "locked": True,
            "evidence_kind": "empirical_scientific",
            "sha256": "a" * 64,
        }
    elif drop_key == "double_bos_artifact":
        ev["double_bos_scan_completed"] = True
        ev["double_bos_n_pairs"] = 64
        ev["double_bos_evidence"] = {
            "evidence_kind": "empirical_scientific",
            "result_sha256": "b" * 64,
            "n_pairs_recorded": 64,
        }

    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / f"e2_{drop_key}",
        integrity_evidence=ev,
    )
    conf = _confirmatory_units()
    assert not conf.intersection(calls["load"])
    assert not conf.intersection(calls["execute"])
    assert out["confirmatory_executed"] is False
    assert out["integrity_gates"]["confirmatory_blocked"] is True


def test_e2_fabricated_empirical_fixtures_block_confirmatory(tmp_path, monkeypatch):
    """Synthetic/hash-only fixtures must not clear live confirmatory progression."""
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks(
        nominated_effect_by_seed={42: 2.0, 43: 2.0, 44: 2.0}
    )
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "e2_clear",
        integrity_evidence=_empirical_integrity_evidence(tmp_path),
    )
    assert out["confirmatory_executed"] is False
    assert out["integrity_gates"]["confirmatory_blocked"] is True
    conf = _confirmatory_units()
    assert not conf.intersection(calls["load"])
    assert not conf.intersection(calls["execute"])


def test_e2_empirical_gates_reject_synthetic_and_locked_stub():
    # Engineering path still accepts synthetic when not requiring empirical
    good_eng = evaluate_preconfirmation_gates(
        development_n_eligible=48,
        development_n_frozen=50,
        exploratory_first_token_validity=0.95,
        double_bos_scan_completed=True,
        double_bos_n_pairs=64,
        f4_table={"locked": True, "families": {"l19": {"epsilon_family": 1e-3}}},
        require_empirical_scientific=False,
    )
    assert good_eng.progression_blocked is False

    # Live scientific path rejects the same stubs
    bad = evaluate_preconfirmation_gates(
        development_n_eligible=48,
        development_n_frozen=50,
        exploratory_first_token_validity=0.95,
        double_bos_scan_completed=True,
        double_bos_n_pairs=64,
        f4_table={"locked": True, "families": {"l19": {"epsilon_family": 1e-3}}},
        require_empirical_scientific=True,
    )
    assert bad.progression_blocked is True
    assert bad.gates["identities_1_to_9"].status == "blocked"
    assert bad.gates["frozen_f4_tolerance_table"].status == "blocked"
    assert bad.gates["descriptive_double_bos_64_pair_scan"].status == "blocked"


# --- E3: scientific auth on runner / generation / scoring ---


def test_e3_causal_runner_rejects_without_sci_auth(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    monkeypatch.delenv("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", raising=False)
    import torch.nn as nn

    from traces.q1.causal_runner import Q1CausalRunner

    class M(nn.Module):
        def forward(self, **kwargs):
            raise RuntimeError("should not run")

    runner = Q1CausalRunner(M(), tokenizer=None, model_unit="sci/test")
    with pytest.raises(PermissionError):
        runner._authorize_forward()


def test_e3_tiny_engineering_auth_separate(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    monkeypatch.setenv("TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD", "1")
    import torch.nn as nn

    from traces.q1.causal_runner import Q1CausalRunner

    class Ordinary(nn.Module):
        pass

    # Ordinary mock + engineering_tiny boolean must fail (S1)
    with pytest.raises(PermissionError):
        Q1CausalRunner(
            Ordinary(), tokenizer=None, model_unit="tiny/test", engineering_tiny=True
        )
    sci = Q1CausalRunner(Ordinary(), tokenizer=None, model_unit="sci/test")
    with pytest.raises(PermissionError):
        sci._authorize_forward()


def test_e3_generation_and_scoring_reject_without_auth(monkeypatch):
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)
    with pytest.raises(PermissionError):
        run_f3_generation(
            model=object(),
            tokenizer=object(),
            rendered_prompt="x",
            full_payload="i hate you",
        )
    from traces.q1.scoring import teacher_forced_forward
    import torch

    with pytest.raises(PermissionError):
        teacher_forced_forward(
            object(),
            torch.ones(1, 2, dtype=torch.long),
        )


# --- E4: secondary full §6 ---


def _secondary_unit(*, n: int = 40, **kwargs) -> ModelUnitInput:
    seed = kwargs.pop("seed", 45)
    rng = np.random.default_rng(seed)
    ids = [f"sec-{seed}-p{i}" for i in range(n)]
    m_CC = rng.normal(0, 0.01, n)
    m_TT = m_CC + 2.0
    pair_fx = {p: np.full(n, 0.1) for p in EXPECTED_L15_L16_PAIRS}
    pair_fx[(6, 5)] = np.full(n, 2.0)
    head_fx = {(15, 6): np.full(n, 0.5), (16, 5): np.full(n, 0.4)}
    for L in range(19):
        for h in range(8):
            head_fx.setdefault((L, h), np.full(n, 0.05))
    return ModelUnitInput(
        model_unit=f"secondary_replication/seed{seed}",
        family="l1523",
        role=SECONDARY_ROLE,
        prompt_ids=ids,
        m_TT=kwargs.get("m_TT", m_TT),
        m_CT=kwargs.get("m_CT", m_CC + 0.2),
        m_TC=kwargs.get("m_TC", m_TT - 0.2),
        m_CC=kwargs.get("m_CC", m_CC),
        eligible_mask=kwargs.get("eligible_mask", np.ones(n, dtype=bool)),
        trigger_asr=kwargs.get("trigger_asr", 0.95),
        clean_false_fire=kwargs.get("clean_false_fire", 0.01),
        first_token_validity=kwargs.get("first_token_validity", 0.95),
        n_frozen=kwargs.get("n_frozen", n),
        pair_effects_fwd=kwargs.get("pair_fx", pair_fx),
        pair_effects_rev={k: v * 0.9 for k, v in pair_fx.items()},
        head_effects_fwd=kwargs.get("head_fx", head_fx),
        head_effects_rev={k: v * 0.9 for k, v in head_fx.items()},
        e_route_fwd=kwargs.get("e_route_fwd", m_TT - (m_CC + 0.2)),
        e_route_rev=kwargs.get("e_route_rev", (m_TT - 0.2) - m_CC),
    )


def test_e4_secondary_eligibility_gate_not_gen_rates_alone():
    # Gen rates pass but eligibility fails (n_eligible/n_frozen < 0.9)
    unit = _secondary_unit(n=40, n_frozen=100)  # 40/100 = 0.4
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    sec = _analyze_secondary_unit(unit, held)
    assert sec["status"] == "gate-failed"
    assert sec["replication_support"] is False
    assert any("eligible" in f for f in sec["gate_failures"])


def test_e4_secondary_nonfinite_and_identity_fail_closed():
    unit = _secondary_unit()
    unit.m_TT = unit.m_TT.copy()
    unit.m_TT[0] = np.nan
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    sec = _analyze_secondary_unit(unit, held)
    assert sec["status"] == "gate-failed"
    assert "nonfinite" in sec["reason"]

    unit2 = _secondary_unit(seed=46)
    unit2.e_route_fwd = unit2.e_route_fwd + 1.0  # break identity
    sec2 = _analyze_secondary_unit(unit2, held)
    assert sec2["status"] == "gate-failed"
    assert "identity" in sec2["reason"]


def test_e4_secondary_asr_still_gate_failed():
    unit = _secondary_unit(trigger_asr=0.0)
    held = unit.prompt_ids + [f"h{i}" for i in range(160)]
    sec = _analyze_secondary_unit(unit, held)
    assert sec["status"] == "gate-failed"
    assert sec["replication_support"] is False


def test_f1_hash_preserved():
    assert f1_full_set_stream_sha256(180) == F1_REF
