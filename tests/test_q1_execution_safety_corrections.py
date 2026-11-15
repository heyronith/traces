"""Execution-safety corrections: exploratory gate, weight tags, ID bind, per-unit provenance.

Synthetic CPU only — no pretrained loads, no model forwards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
from traces.q1.frozen_artifacts import (
    load_ordered_dolly_ids,
    verify_runtime_allocation_ids,
)
from traces.q1.post_load import (
    normalize_weight_manifest_entries,
    validate_weight_manifest_roles,
    verify_post_load,
)
from traces.q1.scientific_execution import (
    ConditionOutputBundle,
    LiveForwardHooks,
    PreflightReport,
    UnitSpec,
    evaluate_live_exploratory_gate,
    expected_frozen_unit_specs,
    run_live_scientific_pipeline,
    verify_held_out_allocation,
    verify_unit_load_provenance,
)

REPO = Path(__file__).resolve().parents[1]


def _frozen_held_out() -> list[int]:
    return load_ordered_dolly_ids(REPO / "results/q1_selection/held_out_instructions.json")


def _frozen_dev() -> list[int]:
    return load_ordered_dolly_ids(
        REPO / "results/q1_selection/development_instructions.json"
    )


def _bundle_for(spec: UnitSpec, n: int = 40) -> ConditionOutputBundle:
    """Synthetic eligible-only margins safe for F1/F2 ratio denominators."""
    seed = spec.seed if spec.seed is not None else 0
    rng = np.random.default_rng(seed + 17)
    ids = [f"{spec.role}-{seed}-p{i}" for i in range(n)]
    mask = np.ones(n, dtype=bool)
    m_CC = rng.normal(0, 0.01, n)
    m_TT = m_CC + 2.0
    m_CT = m_CC + 0.2
    m_TC = m_TT - 0.2
    pair = {
        (h15, h16): rng.normal(0.1, 0.01, n) for h15 in range(8) for h16 in range(8)
    }
    pair[(6, 5)] = pair[(6, 5)] + 1.0
    heads = {(L, h): rng.normal(0.05, 0.01, n) for L in range(19) for h in range(8)}
    heads[(15, 6)] = np.full(n, 0.5)
    heads[(16, 5)] = np.full(n, 0.4)
    return ConditionOutputBundle(
        model_unit=spec.model_unit,
        family=spec.family,
        role=spec.role,
        prompt_ids=ids,
        m_TT=m_TT,
        m_CT=m_CT,
        m_TC=m_TC,
        m_CC=m_CC,
        eligible_mask=mask,
        trigger_asr=0.01 if "adapter_off" in spec.role else 0.95,
        clean_false_fire=0.0,
        first_token_validity=0.95,
        n_frozen=n,
        pair_effects_fwd=pair,
        pair_effects_rev={k: v * 0.9 for k, v in pair.items()},
        head_effects_fwd=heads,
        head_effects_rev={k: v * 0.9 for k, v in heads.items()},
        e_route_fwd=m_TT - m_CT,
        e_route_rev=m_TC - m_CC,
        joint_set_effects_fwd={((15, 6),): heads[(15, 6)]},
        joint_set_effects_rev={((15, 6),): heads[(15, 6)] * 0.9},
        catalog_condition_ids=["single_head|forward"],
        eligible_prompt_ids=list(ids),
    )


def _pass_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")

    def _ok(*_a: Any, **_k: Any) -> PreflightReport:
        return PreflightReport(ok=True, failures=[], checks={"test": True})

    monkeypatch.setattr(
        "traces.q1.scientific_execution.run_scientific_preflight", _ok
    )
    monkeypatch.setattr(
        "traces.q1.scientific_execution.require_scientific_authorization",
        lambda: None,
    )


def _spy_hooks() -> tuple[LiveForwardHooks, dict[str, list[str]]]:
    calls: dict[str, list[str]] = {
        "load": [],
        "donors": [],
        "execute": [],
    }

    def load(spec: UnitSpec) -> dict[str, Any]:
        calls["load"].append(spec.model_unit)
        return {
            "model": object(),
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
        return {
            "n_donor_caches": 8,
            "clean_references_frozen": True,
            "instruction_ids": [f"p{i}" for i in range(8)],
        }

    def execute(spec: UnitSpec, _model: Any, _donors: Any) -> tuple:
        calls["execute"].append(spec.model_unit)
        b = _bundle_for(spec)
        return b, b

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


# --- 1. Exploratory stop before confirmatory callbacks ---


def test_gate_blocks_on_stop_ranks():
    specs = [s for s in expected_frozen_unit_specs() if s.role == "exploratory_dev"]
    gate = evaluate_live_exploratory_gate(
        exploratory_specs=specs, exploratory_ranks=[11, 12, 1]
    )
    assert gate["triggered"] is True
    assert gate["confirmatory_blocked"] is True
    assert gate["fabricated_ranks"] is False


def test_gate_blocks_on_missing_ranks_without_fabrication():
    specs = [s for s in expected_frozen_unit_specs() if s.role == "exploratory_dev"]
    gate = evaluate_live_exploratory_gate(
        exploratory_specs=specs, exploratory_ranks=None
    )
    assert gate["confirmatory_blocked"] is True
    assert gate["fabricated_ranks"] is False
    assert gate["ranks"] is None


def test_spy_zero_confirmatory_when_stop_triggered(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "stop",
        exploratory_ranks=[15, 20, 3],  # ≥2 worse than 10
    )
    conf = _confirmatory_units()
    assert not conf.intersection(calls["load"])
    assert not conf.intersection(calls["donors"])
    assert not conf.intersection(calls["execute"])
    assert out["confirmatory_executed"] is False
    assert out["exploratory_gate"]["triggered"] is True
    # Exactly three exploratory loads
    assert len(calls["load"]) == 3
    assert all("exploratory" in u for u in calls["load"])


def test_spy_zero_confirmatory_when_ranks_missing(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "missing_ranks",
        exploratory_ranks=None,
    )
    conf = _confirmatory_units()
    assert len(calls["load"]) == 3
    assert not conf.intersection(calls["load"])
    assert not conf.intersection(calls["execute"])
    assert out["confirmatory_executed"] is False
    assert out["exploratory_gate"]["fabricated_ranks"] is False


# --- 2. Role-tagged weight hashing ---


def test_normalize_role_tagged_keys(tmp_path):
    base = tmp_path / "model.safetensors"
    adapter = tmp_path / "adapter_model.safetensors"
    base.write_bytes(b"base-bytes")
    adapter.write_bytes(b"adapter-bytes")
    norm = normalize_weight_manifest_entries(
        {
            f"base:{base}": "a" * 64,
            f"adapter:{adapter}": "b" * 64,
        }
    )
    assert norm[f"base:{base}"][0] == str(base)
    assert norm[f"adapter:{adapter}"][0] == str(adapter)


def test_tagged_manifest_pass_and_altered_fail(tmp_path):
    base = tmp_path / "model.safetensors"
    adapter = tmp_path / "adapter_model.safetensors"
    base.write_bytes(b"base-ok")
    adapter.write_bytes(b"adapter-ok")
    b_digest = hashlib.sha256(base.read_bytes()).hexdigest()
    a_digest = hashlib.sha256(adapter.read_bytes()).hexdigest()

    class FakeCfg:
        _attn_implementation = "eager"
        _commit_hash = PINNED_BASE_REVISION

    class FakePeft:
        target_modules = ["q_proj", "v_proj"]
        revision = PINNED_ADAPTER_REVISION

    class FakeModel:
        config = FakeCfg()
        peft_config = {"default": FakePeft()}

        def parameters(self):
            import torch

            yield torch.ones(2, dtype=torch.bfloat16)

        def named_parameters(self):
            import torch

            yield (
                "layers.19.self_attn.q_proj.lora_A.default.weight",
                torch.ones(2, dtype=torch.bfloat16),
            )
            yield (
                "layers.19.self_attn.v_proj.lora_B.default.weight",
                torch.ones(2, dtype=torch.bfloat16),
            )

    class FakeTok:
        is_fast = False

    good = verify_post_load(
        FakeModel(),
        FakeTok(),
        expected_base_revision=PINNED_BASE_REVISION,
        expected_adapter_revision=PINNED_ADAPTER_REVISION,
        expected_weight_file_sha256={
            f"base:{base}": b_digest,
            f"adapter:{adapter}": a_digest,
        },
        expected_tokenizer_class="FakeTok",
        expected_tokenizer_is_fast=False,
        expected_adapter_targets=["q_proj", "v_proj"],
        expected_adapted_layers=[19],
    )
    assert good.ok is True

    # Alter file → digest mismatch
    base.write_bytes(b"base-TAMPERED")
    bad = verify_post_load(
        FakeModel(),
        FakeTok(),
        expected_base_revision=PINNED_BASE_REVISION,
        expected_adapter_revision=PINNED_ADAPTER_REVISION,
        expected_weight_file_sha256={
            f"base:{base}": b_digest,
            f"adapter:{adapter}": a_digest,
        },
        expected_tokenizer_class="FakeTok",
        expected_tokenizer_is_fast=False,
        expected_adapter_targets=["q_proj", "v_proj"],
        expected_adapted_layers=[19],
    )
    assert bad.ok is False
    assert any("SHA mismatch" in f or "sha" in f.lower() for f in bad.failures)


def test_reject_missing_wrong_unbound_ambiguous(tmp_path):
    base = tmp_path / "model.safetensors"
    adapter = tmp_path / "adapter_model.safetensors"
    base.write_bytes(b"b")
    adapter.write_bytes(b"a")
    # Missing resolved coverage
    fails = validate_weight_manifest_roles(
        {f"base:{base}": "a" * 64},
        resolved_base_files=[str(base)],
        resolved_adapter_files=[str(adapter)],
        require_resolved_inventories=True,
    )
    assert any("adapter" in f for f in fails)
    # Unbound extra
    fails2 = validate_weight_manifest_roles(
        {
            f"base:{base}": "a" * 64,
            f"adapter:{adapter}": "b" * 64,
            "base:/not/resolved.bin": "c" * 64,
        },
        resolved_base_files=[str(base)],
        resolved_adapter_files=[str(adapter)],
        require_resolved_inventories=True,
    )
    assert any("unbound" in f for f in fails2)
    # Ambiguous empty path after tag
    with pytest.raises(ValueError, match="empty path"):
        normalize_weight_manifest_entries({"base:": "a" * 64})


# --- 3. Bind runtime IDs to frozen allocation ---


def test_allocation_accepts_exact_frozen_ids():
    rep = verify_runtime_allocation_ids(
        REPO, held_out_ids=_frozen_held_out(), development_ids=_frozen_dev()
    )
    assert rep["ok"] is True


def test_allocation_rejects_reordered():
    held = _frozen_held_out()
    reordered = list(reversed(held))
    rep = verify_runtime_allocation_ids(REPO, held_out_ids=reordered)
    assert rep["ok"] is False
    assert any("reorder" in f or "match" in f for f in rep["failures"])


def test_allocation_rejects_substituted():
    held = _frozen_held_out()
    bad = list(held)
    bad[0] = 999999
    rep = verify_runtime_allocation_ids(REPO, held_out_ids=bad)
    assert rep["ok"] is False


def test_allocation_rejects_duplicated():
    held = _frozen_held_out()
    bad = list(held)
    bad[1] = bad[0]
    rep = verify_runtime_allocation_ids(REPO, held_out_ids=bad)
    assert rep["ok"] is False
    assert any("duplicate" in f for f in rep["failures"])


def test_allocation_rejects_truncated():
    held = _frozen_held_out()[:199]
    rep = verify_runtime_allocation_ids(REPO, held_out_ids=held)
    assert rep["ok"] is False


def test_allocation_rejects_non_dolly_instruction_ids():
    held = [f"instr-{i}" for i in range(200)]
    rep = verify_runtime_allocation_ids(REPO, held_out_ids=held)
    assert rep["ok"] is False


def test_held_out_binding_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="frozen allocation"):
        verify_held_out_allocation(
            list(reversed(_frozen_held_out())),
            repo_root=REPO,
        )


# --- 4. Per-unit provenance (no first-model reuse) ---


def test_per_unit_provenance_rejects_identity_mismatch():
    spec = UnitSpec(
        model_unit="exploratory_l1523_seed42",
        family="l1523",
        role="exploratory_dev",
        seed=42,
    )
    # Provenance from a *different* unit (simulating reuse of first model)
    wrong = {
        "model_unit": "exploratory_l1523_seed43",
        "family": "l1523",
        "role": "exploratory_dev",
        "seed": 43,
        "base_revision": PINNED_BASE_REVISION,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "attn_implementation": "eager",
        "torch_dtype": "torch.bfloat16",
        "weight_file_sha256": {"base:/x": "a" * 64},
    }
    rep = verify_unit_load_provenance(spec, wrong)
    assert rep["ok"] is False
    assert any("model_unit" in f or "seed" in f for f in rep["failures"])


def test_live_pipeline_audits_each_unit_independently(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks()
    # Clear gate so confirmatory would run — but use ranks that pass
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "full",
        exploratory_ranks=[1, 2, 3],
    )
    assert out["confirmatory_executed"] is True
    assert len(out["per_unit"]) == len(expected_frozen_unit_specs())
    units_seen = [a["model_unit"] for a in out["per_unit"]]
    assert len(units_seen) == len(set(units_seen))
    for a in out["per_unit"]:
        assert a["provenance"]["ok"] is True
        assert a["provenance"]["model_unit"] == a["model_unit"]
        assert a["donors"]["ok"] is True
    # Spy saw confirmatory loads only after exploratory
    conf = _confirmatory_units()
    assert conf.issubset(set(calls["load"]))
    # First three loads are exploratory
    assert all("exploratory" in u for u in calls["load"][:3])


def test_live_rejects_reused_first_unit_provenance(tmp_path, monkeypatch):
    """If load always echoes seed42 identity, later units must fail."""
    _pass_preflight(monkeypatch)
    first_spec = expected_frozen_unit_specs()[0]
    calls: list[str] = []

    def load(spec: UnitSpec) -> dict[str, Any]:
        calls.append(spec.model_unit)
        # Always return first unit's identity (bug under test)
        return {
            "model": object(),
            "model_unit": first_spec.model_unit,
            "family": first_spec.family,
            "role": first_spec.role,
            "seed": first_spec.seed,
            "base_revision": PINNED_BASE_REVISION,
            "adapter_revision": PINNED_ADAPTER_REVISION,
            "weight_file_sha256": {"base:/x": "a" * 64},
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }

    def donors(spec: UnitSpec, _m: Any) -> dict[str, Any]:
        return {
            "n_donor_caches": 4,
            "clean_references_frozen": True,
            "instruction_ids": ["p0"],
        }

    def execute(spec: UnitSpec, _m: Any, _d: Any) -> tuple:
        b = _bundle_for(spec)
        return b, b

    with pytest.raises(RuntimeError, match="provenance failed"):
        run_live_scientific_pipeline(
            repo_root=REPO,
            held_out_ids_allocation_order=_frozen_held_out(),
            development_ids=_frozen_dev(),
            hooks=LiveForwardHooks(
                load_model_for_unit=load,
                capture_donors_for_unit=donors,
                execute_unit=execute,
            ),
            out_dir=tmp_path / "reuse",
            exploratory_ranks=[1, 2, 3],
        )
    # Second exploratory unit triggers the mismatch
    assert len(calls) >= 2
