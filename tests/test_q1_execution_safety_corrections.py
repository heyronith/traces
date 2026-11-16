"""Execution-safety corrections: bound ranks, PEFT adapter-off, weight inventories.

Synthetic CPU only — no pretrained loads, no model forwards.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION
from traces.q1.component_sets import EXPECTED_L15_L16_PAIRS
from traces.q1.frozen_artifacts import (
    load_ordered_dolly_ids,
    verify_runtime_allocation_ids,
)
from traces.q1.loader import validate_loader_policy_inputs
from traces.q1.post_load import (
    normalize_weight_manifest_entries,
    require_bound_weight_inventories,
    resolve_weight_files_from_model,
    validate_weight_manifest_roles,
    verify_adapter_off_behavior,
    verify_post_load,
)
from traces.q1.pipeline_adapters import mock_condition_records_for_unit
from traces.q1.scientific_execution import (
    NOMINATED_PAIR_KEY,
    ConditionOutputBundle,
    LiveForwardHooks,
    PreflightReport,
    UnitSpec,
    bind_exploratory_ranks_from_results,
    evaluate_live_exploratory_gate,
    expected_frozen_unit_specs,
    run_live_scientific_pipeline,
    stash_mock_observations_on_donors,
    verify_held_out_allocation,
    verify_unit_load_provenance,
)
from traces.q1.workload import N_DEV

REPO = Path(__file__).resolve().parents[1]


def _frozen_held_out() -> list[int]:
    return load_ordered_dolly_ids(REPO / "results/q1_selection/held_out_instructions.json")


def _frozen_dev() -> list[int]:
    return load_ordered_dolly_ids(
        REPO / "results/q1_selection/development_instructions.json"
    )


def _bundle_for(
    spec: UnitSpec,
    *,
    prompt_ids: list[Any] | None = None,
    nominated_effect: float = 2.0,
    other_effect: float = 0.1,
) -> ConditionOutputBundle:
    """Synthetic bundle; control nominated-pair rank via effect magnitudes."""
    seed = spec.seed if spec.seed is not None else 0
    rng = np.random.default_rng(seed + 17)
    if prompt_ids is None:
        if spec.role == "exploratory_dev":
            prompt_ids = list(_frozen_dev())
        else:
            prompt_ids = [f"{spec.role}-{seed}-p{i}" for i in range(40)]
    n = len(prompt_ids)
    ids = [str(x) for x in prompt_ids]
    mask = np.ones(n, dtype=bool)
    m_CC = rng.normal(0, 0.01, n)
    m_TT = m_CC + 2.0
    m_CT = m_CC + 0.2
    m_TC = m_TT - 0.2
    pair = {p: np.full(n, other_effect) for p in EXPECTED_L15_L16_PAIRS}
    pair[NOMINATED_PAIR_KEY] = np.full(n, nominated_effect)
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
        bos_convention="single",
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
        b = _bundle_for(spec, nominated_effect=nom, other_effect=0.1)
        ids = list(donors_ev.get("instruction_ids") or [f"p{i}" for i in range(8)])
        recs = mock_condition_records_for_unit(spec, ids, nominated_effect=nom)
        stash_mock_observations_on_donors(
            donors_ev, spec=spec, records=recs, prompt_ids=ids
        )
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


# --- 1. Bound exploratory ranks ---


def test_gate_rejects_unbound_caller_ranks_without_results():
    specs = [s for s in expected_frozen_unit_specs() if s.role == "exploratory_dev"]
    gate = evaluate_live_exploratory_gate(
        exploratory_specs=specs,
        exploratory_units=None,
        development_ids=_frozen_dev(),
        exploratory_ranks=[1, 2, 3],
    )
    assert gate["confirmatory_blocked"] is True
    assert gate["caller_ranks_rejected"] is True
    assert gate["fabricated_ranks"] is False
    assert gate["ranks"] is None


def test_spy_zero_confirmatory_when_bound_ranks_trigger_stop(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    # Low nominated effect → rank near bottom (>10)
    hooks, calls = _spy_hooks(
        nominated_effect_by_seed={42: -1.0, 43: -1.0, 44: 2.0}
    )
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "stop",
        exploratory_ranks=None,
    )
    conf = _confirmatory_units()
    assert not conf.intersection(calls["load"])
    assert not conf.intersection(calls["execute"])
    assert out["confirmatory_executed"] is False
    assert out["exploratory_gate"]["triggered"] is True
    assert out["exploratory_gate"]["ok"] is True
    assert len(calls["load"]) == 3


def test_spy_adversarial_caller_passing_ranks_ignored(tmp_path, monkeypatch):
    """Caller supplies [1,2,3] but actual results trigger stop → still block."""
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks(
        nominated_effect_by_seed={42: -1.0, 43: -1.0, 44: -1.0}
    )
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "adversarial",
        exploratory_ranks=[1, 2, 3],  # unbound / disagreeing
    )
    conf = _confirmatory_units()
    assert not conf.intersection(calls["load"])
    assert out["confirmatory_executed"] is False
    assert out["exploratory_gate"]["caller_ranks_rejected"] is True


def test_spy_zero_confirmatory_when_incomplete_comparator(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    calls: dict[str, list[str]] = {"load": [], "donors": [], "execute": []}

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
            "adapter_revision": PINNED_ADAPTER_REVISION,
            "weight_file_sha256": {"base:/x": "a" * 64},
            "attn_implementation": "eager",
            "torch_dtype": "torch.bfloat16",
        }

    def donors(spec: UnitSpec, _m: Any) -> dict[str, Any]:
        calls["donors"].append(spec.model_unit)
        return {
            "n_donor_caches": N_DEV,
            "clean_references_frozen": True,
            "instruction_ids": [str(x) for x in _frozen_dev()],
        }

    def execute(spec: UnitSpec, _m: Any, donors_ev: Any) -> tuple:
        calls["execute"].append(spec.model_unit)
        b = _bundle_for(spec, nominated_effect=2.0)
        # Incomplete comparator population
        b.pair_effects_fwd = {NOMINATED_PAIR_KEY: b.pair_effects_fwd[NOMINATED_PAIR_KEY]}
        b.pair_effects_rev = {NOMINATED_PAIR_KEY: b.pair_effects_rev[NOMINATED_PAIR_KEY]}
        ids = list(donors_ev.get("instruction_ids") or [f"p{i}" for i in range(8)])
        recs = mock_condition_records_for_unit(spec, ids, nominated_effect=2.0)
        stash_mock_observations_on_donors(
            donors_ev, spec=spec, records=recs, prompt_ids=ids
        )
        return b, b

    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=LiveForwardHooks(
            load_model_for_unit=load,
            capture_donors_for_unit=donors,
            execute_unit=execute,
        ),
        out_dir=tmp_path / "incomplete",
    )
    assert out["confirmatory_executed"] is False
    assert not _confirmatory_units().intersection(calls["load"])
    assert out["exploratory_gate"]["ok"] is False


def _empirical_integrity_evidence(tmp_path: Path) -> dict[str, Any]:
    """Persisted fabricated artifacts with independently checkable hashes."""
    from traces.q1.empirical_evidence import (
        write_fabricated_double_bos_artifact,
        write_fabricated_f4_artifact,
        write_fabricated_identity_artifact,
    )

    id_p, id_s = write_fabricated_identity_artifact(tmp_path / "identity.json")
    f4_p, f4_s = write_fabricated_f4_artifact(tmp_path / "f4.json")
    db_p, db_s = write_fabricated_double_bos_artifact(tmp_path / "double_bos.json")
    return {
        "identity_artifact": {"artifact_path": str(id_p), "sha256": id_s},
        "f4_artifact": {"artifact_path": str(f4_p), "sha256": f4_s},
        "double_bos_artifact": {"artifact_path": str(db_p), "sha256": db_s},
    }


def test_bound_ranks_clear_but_integrity_provenance_still_blocks(tmp_path, monkeypatch):
    """Ranks may clear from exploratory results; fabricated integrity still BLOCKS."""
    _pass_preflight(monkeypatch)
    hooks, calls = _spy_hooks(
        nominated_effect_by_seed={42: 2.0, 43: 2.0, 44: 2.0}
    )
    out = run_live_scientific_pipeline(
        repo_root=REPO,
        held_out_ids_allocation_order=_frozen_held_out(),
        development_ids=_frozen_dev(),
        hooks=hooks,
        out_dir=tmp_path / "clear",
        integrity_evidence=_empirical_integrity_evidence(tmp_path),
    )
    assert out["exploratory_gate"]["triggered"] is False
    assert out["exploratory_gate"]["ranks"] == [1, 1, 1]
    assert out["integrity_gates"]["confirmatory_blocked"] is True
    assert out["confirmatory_executed"] is False
    conf = _confirmatory_units()
    assert not conf.intersection(calls["load"])
    assert not conf.intersection(calls["execute"])
    assert all("exploratory" in u for u in calls["load"])
    assert len(calls["load"]) == 3


# --- 2. PEFT 0.19.1 adapter-off ---


def test_adapter_off_against_pinned_peft_api():
    import torch
    import torch.nn as nn
    from peft import LoraConfig, PeftModel, get_peft_model

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    base = Tiny()
    cfg = LoraConfig(r=2, lora_alpha=4, target_modules=["linear"], bias="none")
    model = get_peft_model(base, cfg)
    assert isinstance(model, PeftModel)
    assert hasattr(model, "disable_adapter")
    assert hasattr(model.base_model, "disable_adapter_layers")
    assert hasattr(model.base_model, "enable_adapter_layers")
    # PeftModel itself does NOT expose disable_adapter_layers
    assert not hasattr(model, "disable_adapter_layers") or not callable(
        getattr(type(model), "disable_adapter_layers", None)
    )

    rep = verify_adapter_off_behavior(model)
    assert rep["ok"] is True
    assert rep["restored"] is True
    assert rep["verification_kind"] == "mechanical_state_switch"
    assert rep["numerical_adapter_off_validated"] is False
    assert rep["peft_api"] is not None
    assert "disable_adapter" in (rep["peft_api"] or "")


def test_adapter_off_rejects_missing_api():
    class Empty:
        pass

    rep = verify_adapter_off_behavior(Empty())
    assert rep["ok"] is False
    assert rep["numerical_adapter_off_validated"] is False


def test_adapter_off_legacy_direct_methods_still_restore():
    class Flaky:
        def __init__(self) -> None:
            self.enabled = True

        def disable_adapter_layers(self) -> None:
            self.enabled = False
            raise RuntimeError("boom")

        def enable_adapter_layers(self) -> None:
            self.enabled = True

    r = verify_adapter_off_behavior(Flaky())
    assert r["restored"] is True
    assert r["ok"] is False


# --- 3. Weight inventory binding ---


def test_empty_inventory_fails_closed(tmp_path):
    class Bare:
        config = type("C", (), {"_name_or_path": str(tmp_path / "missing")})()

    resolved = resolve_weight_files_from_model(Bare())
    assert resolved["base"] == []
    assert resolved["adapter"] == []
    bind = require_bound_weight_inventories(
        resolved,
        {
            f"base:{tmp_path / 'a.safetensors'}": "a" * 64,
            f"adapter:{tmp_path / 'b.safetensors'}": "b" * 64,
        },
    )
    assert bind["ok"] is False
    assert any("base weight inventory empty" in f for f in bind["failures"])
    assert any("adapter weight inventory empty" in f for f in bind["failures"])


def test_valid_fabricated_snapshots_bind(tmp_path):
    base_dir = tmp_path / "base_snap"
    ad_dir = tmp_path / "adapter_snap"
    base_dir.mkdir()
    ad_dir.mkdir()
    base_f = base_dir / "model.safetensors"
    ad_f = ad_dir / "adapter_model.safetensors"
    base_f.write_bytes(b"base-snap")
    ad_f.write_bytes(b"adapter-snap")
    b_digest = hashlib.sha256(base_f.read_bytes()).hexdigest()
    a_digest = hashlib.sha256(ad_f.read_bytes()).hexdigest()

    class M:
        pass

    model = M()
    model._q1_base_weight_dir = str(base_dir)
    model._q1_adapter_weight_dir = str(ad_dir)
    resolved = resolve_weight_files_from_model(model)
    assert str(base_f) in resolved["base"]
    assert str(ad_f) in resolved["adapter"]
    bind = require_bound_weight_inventories(
        resolved,
        {f"base:{base_f}": b_digest, f"adapter:{ad_f}": a_digest},
    )
    assert bind["ok"] is True


def test_invalid_fabricated_snapshot_wrong_digest_and_unrelated(tmp_path):
    base_dir = tmp_path / "base_snap"
    ad_dir = tmp_path / "adapter_snap"
    base_dir.mkdir()
    ad_dir.mkdir()
    base_f = base_dir / "model.safetensors"
    ad_f = ad_dir / "adapter_model.safetensors"
    base_f.write_bytes(b"base-snap")
    ad_f.write_bytes(b"adapter-snap")
    unrelated = tmp_path / "unrelated.safetensors"
    unrelated.write_bytes(b"other")
    u_digest = hashlib.sha256(unrelated.read_bytes()).hexdigest()
    a_digest = hashlib.sha256(ad_f.read_bytes()).hexdigest()

    class M:
        pass

    model = M()
    model._q1_base_weight_dir = str(base_dir)
    model._q1_adapter_weight_dir = str(ad_dir)
    resolved = resolve_weight_files_from_model(model)
    # Manifest covers unrelated file with consistent self-hash, not inventory
    bind = require_bound_weight_inventories(
        resolved,
        {
            f"base:{unrelated}": u_digest,
            f"adapter:{ad_f}": a_digest,
        },
    )
    assert bind["ok"] is False
    assert any(
        "missing resolved base" in f or "unbound" in f or "no manifest digest" in f
        for f in bind["failures"]
    )


def test_normalize_role_tagged_keys(tmp_path):
    base = tmp_path / "model.safetensors"
    adapter = tmp_path / "adapter_model.safetensors"
    base.write_bytes(b"base-bytes")
    adapter.write_bytes(b"adapter-bytes")
    norm = normalize_weight_manifest_entries(
        {f"base:{base}": "a" * 64, f"adapter:{adapter}": "b" * 64}
    )
    assert norm[f"base:{base}"][0] == str(base)


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


# --- Allocation binding (retained) ---


def test_allocation_accepts_exact_frozen_ids():
    rep = verify_runtime_allocation_ids(
        REPO, held_out_ids=_frozen_held_out(), development_ids=_frozen_dev()
    )
    assert rep["ok"] is True


def test_allocation_rejects_reordered_substituted_duplicated_truncated():
    held = _frozen_held_out()
    assert verify_runtime_allocation_ids(REPO, held_out_ids=list(reversed(held)))[
        "ok"
    ] is False
    bad = list(held)
    bad[0] = 999999
    assert verify_runtime_allocation_ids(REPO, held_out_ids=bad)["ok"] is False
    dup = list(held)
    dup[1] = dup[0]
    assert verify_runtime_allocation_ids(REPO, held_out_ids=dup)["ok"] is False
    assert verify_runtime_allocation_ids(REPO, held_out_ids=held[:199])["ok"] is False


def test_held_out_binding_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="frozen allocation"):
        verify_held_out_allocation(
            list(reversed(_frozen_held_out())),
            repo_root=REPO,
        )


# --- Per-unit provenance (retained) ---


def test_per_unit_provenance_rejects_identity_mismatch():
    spec = UnitSpec(
        model_unit="exploratory_l1523_seed42",
        family="l1523",
        role="exploratory_dev",
        seed=42,
    )
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


def test_live_rejects_reused_first_unit_provenance(tmp_path, monkeypatch):
    _pass_preflight(monkeypatch)
    first_spec = expected_frozen_unit_specs()[0]
    calls: list[str] = []

    def load(spec: UnitSpec) -> dict[str, Any]:
        calls.append(spec.model_unit)
        return {
            "model": object(),
            "tokenizer": object(),
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
            "n_donor_caches": N_DEV,
            "clean_references_frozen": True,
            "instruction_ids": [str(x) for x in _frozen_dev()],
        }

    def execute(spec: UnitSpec, _m: Any, donors_ev: Any) -> tuple:
        b = _bundle_for(spec)
        ids = list(donors_ev.get("instruction_ids") or [f"p{i}" for i in range(8)])
        recs = mock_condition_records_for_unit(spec, ids)
        stash_mock_observations_on_donors(
            donors_ev, spec=spec, records=recs, prompt_ids=ids
        )
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
        )
    assert len(calls) >= 2
