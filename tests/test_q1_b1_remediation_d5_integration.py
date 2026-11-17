"""D5-A1 / D5-B1 / D5-A2 integration remediation — no pretrained / no forwards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traces.q1.checkpoint_provenance import (
    bind_unit_checkpoint_provenance,
    engineering_bind_unit_checkpoint_provenance,
)
from traces.q1.checkpoint_trust import (
    ENGINEERING_CHECKPOINT_TRUST_ENV,
    PINNED_CHECKPOINT_MANIFEST_SHA256,
    authenticate_engineering_checkpoint_manifest,
    authenticate_production_checkpoint_manifest,
    parse_safetensors_index,
    unit_digests_from_auth,
)
from traces.q1.loader import load_q1_scientific_model
from traces.q1.production_pipeline import ProductionHookConfig, build_production_live_hooks
from traces.q1.scientific_execution import (
    LiveForwardHooks,
    UnitSpec,
    _execute_live_unit,
    run_live_scientific_pipeline,
    verify_unit_load_provenance,
)
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256
from traces.model_loader import PINNED_ADAPTER_REVISION, PINNED_BASE_REVISION

from q1_checkpoint_test_support import (
    install_temporary_production_pin,
    make_sharded_ten_unit_fixture,
    write_sharded_base,
)

REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def eng_trust(monkeypatch):
    monkeypatch.setenv(ENGINEERING_CHECKPOINT_TRUST_ENV, "1")
    yield
    monkeypatch.delenv(ENGINEERING_CHECKPOINT_TRUST_ENV, raising=False)


def _spec(**kw) -> UnitSpec:
    d = dict(
        model_unit="exploratory_l19_seed42",
        family="l19",
        role="exploratory_dev",
        seed=42,
    )
    d.update(kw)
    return UnitSpec(**d)


# --- D5-A1 ------------------------------------------------------------------


def test_d5a1_explicit_repo_root_required_not_inferred(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    base = tmp_path / "external_base"
    adapter = tmp_path / "external_adapter"
    base.mkdir()
    adapter.mkdir()
    with pytest.raises(RuntimeError, match="trusted executor repo_root|D5-A1"):
        load_q1_scientific_model(
            subfolder="l19/seed42",
            expected_adapter_targets=["q_proj"],
            expected_adapted_layers=[19],
            base_weight_snapshot_dir=str(base),
            adapter_weight_snapshot_dir=str(adapter),
            repo_root=None,
        )
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_d5a1_external_snapshots_with_temp_pin(monkeypatch, tmp_path, eng_trust):
    """Valid synthetic pin works with checkpoints outside the repo (D5-A1)."""
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    # Executor repo (trusted) separate from external snapshots; temp pin overrides.
    repo = tmp_path / "executor_repo"
    (repo / "src" / "traces").mkdir(parents=True)
    (repo / "results" / "q1_implementation_prep").mkdir(parents=True)
    external = tmp_path / "external_snapshots"
    fix = make_sharded_ten_unit_fixture(external)
    digest = install_temporary_production_pin(monkeypatch, repo, fix["payload"])
    assert digest
    auth = authenticate_production_checkpoint_manifest(repo_root=repo)
    assert auth.production_unlocked is True
    # Bind against external dirs using production auth digests
    digests = unit_digests_from_auth(auth, subfolder="l19/seed42")
    bind = bind_unit_checkpoint_provenance(
        family="l19",
        seed=42,
        subfolder="l19/seed42",
        base_snapshot_dir=fix["base"],
        adapter_snapshot_dir=fix["adapter"],
        expected_sha256_by_relpath=digests,
        require_section11_tokenizer_pins=False,
        require_authoritative_trust=True,
        trust_auth=auth,
    )
    assert bind["ok"] is True
    assert len(bind["base_shards"]) == 3

    # Incorrect pin blocks the same layout
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        "0" * 64,
    )
    auth_bad = authenticate_production_checkpoint_manifest(repo_root=repo)
    assert auth_bad.production_unlocked is False
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_d5a1_production_hooks_pass_repo_root(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    captured: dict[str, Any] = {}

    def fake_load(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop-after-capture")

    with patch("traces.q1.loader.load_q1_scientific_model", side_effect=fake_load), patch(
        "traces.q1.checkpoint_trust.authenticate_production_checkpoint_manifest",
        return_value=MagicMock(
            production_unlocked=True, engineering_only=False, failures=[], ok=True
        ),
    ):
        hooks = build_production_live_hooks(
            ProductionHookConfig(
                repo_root=REPO,
                base_weight_snapshot_dir=str(tmp_path / "base"),
                adapter_weight_snapshot_dir=str(tmp_path / "adapter"),
            )
        )
        (tmp_path / "base").mkdir()
        (tmp_path / "adapter").mkdir()
        with pytest.raises(RuntimeError, match="stop-after-capture"):
            hooks.load_model_for_unit(_spec())
    assert captured.get("repo_root") == REPO
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


# --- D5-B1 ------------------------------------------------------------------


def test_d5b1_valid_three_shard_layout(eng_trust, tmp_path):
    fix = make_sharded_ten_unit_fixture(tmp_path)
    auth = authenticate_engineering_checkpoint_manifest(fix["payload"])
    assert auth.ok is True
    bind = engineering_bind_unit_checkpoint_provenance(
        family="l19",
        seed=42,
        subfolder="l19/seed42",
        base_snapshot_dir=fix["base"],
        adapter_snapshot_dir=fix["adapter"],
        engineering_manifest=fix["payload"],
        require_section11_tokenizer_pins=False,
    )
    assert bind["ok"] is True
    assert bind["base_shards"] == [
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]


def test_d5b1_missing_shard_fails(eng_trust, tmp_path):
    fix = make_sharded_ten_unit_fixture(tmp_path)
    (fix["base"] / "model-00002-of-00003.safetensors").unlink()
    with pytest.raises(RuntimeError, match="missing|SHA|shard|D5-B1"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )


def test_d5b1_extra_shard_fails(eng_trust, tmp_path):
    fix = make_sharded_ten_unit_fixture(tmp_path)
    (fix["base"] / "model-00004-of-00004.safetensors").write_bytes(b"EXTRA")
    with pytest.raises(RuntimeError, match="unlisted|extra|D5-B1"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )


def test_d5b1_altered_shard_fails(eng_trust, tmp_path):
    fix = make_sharded_ten_unit_fixture(tmp_path)
    p = fix["base"] / "model-00001-of-00003.safetensors"
    p.write_bytes(b"TAMPERED-SHARD")
    with pytest.raises(RuntimeError, match="SHA|mismatch"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )


def test_d5b1_index_referencing_different_shard_fails(eng_trust, tmp_path):
    fix = make_sharded_ten_unit_fixture(tmp_path)
    # Rewrite index to reference a shard not in the authoritative manifest
    bad_index = {
        "weight_map": {"x": "model-99999-of-00003.safetensors"},
        "metadata": {},
    }
    # Keep digest in payload pointing at original index — bind will see SHA mismatch
    # OR update bytes without updating manifest digest
    (fix["base"] / "model.safetensors.index.json").write_bytes(
        json.dumps(bad_index).encode()
    )
    with pytest.raises(RuntimeError, match="SHA|mismatch|missing|D5-B1"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )


def test_d5b1_parse_rejects_path_escape_in_weight_map():
    with pytest.raises(RuntimeError, match="escape|basename"):
        parse_safetensors_index({"weight_map": {"a": "../evil.safetensors"}})


# --- D5-A2 ------------------------------------------------------------------


def test_d5a2_live_pipeline_blocks_without_pin(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        None,
    )
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    from traces.q1.frozen_artifacts import load_ordered_dolly_ids
    from traces.q1.scientific_execution import PreflightReport

    held = load_ordered_dolly_ids(REPO / "results/q1_selection/held_out_instructions.json")
    hooks = LiveForwardHooks(
        load_model_for_unit=lambda s: (_ for _ in ()).throw(RuntimeError("should-not-load")),
        capture_donors_for_unit=lambda s, m: {},
        execute_unit=lambda s, m, d: {},
    )
    # Clear preflight so D5-A2 executor checkpoint trust is the blocking gate
    with patch(
        "traces.q1.scientific_execution.run_scientific_preflight",
        return_value=PreflightReport(ok=True, failures=[], checks={}),
    ):
        with pytest.raises(RuntimeError, match="checkpoint trust|D5-A2|unset"):
            run_live_scientific_pipeline(
                repo_root=REPO,
                held_out_ids_allocation_order=held,
                hooks=hooks,
                out_dir=tmp_path / "out",
            )
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_d5a2_engineering_trust_rejected_on_live_path(monkeypatch, tmp_path, eng_trust):
    fix = make_sharded_ten_unit_fixture(tmp_path)
    eng = authenticate_engineering_checkpoint_manifest(fix["payload"])
    assert eng.ok is True
    assert eng.production_unlocked is False
    assert eng.engineering_only is True
    spec = _spec()
    prov = {
        "model_unit": spec.model_unit,
        "family": spec.family,
        "role": spec.role,
        "seed": spec.seed,
        "base_revision": PINNED_BASE_REVISION,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "attn_implementation": "eager",
        "torch_dtype": "torch.bfloat16",
        "weight_file_sha256": {"base:x": "a" * 64},
        "adapter_subfolder": "l19/seed42",
        "checkpoint_trust": eng.to_dict(),
        "base_weight_snapshot_dir": str(fix["base"]),
        "adapter_weight_snapshot_dir": str(fix["adapter"]),
    }
    # Executor auth must be production_unlocked — engineering auth fails
    rep = verify_unit_load_provenance(
        spec, prov, executor_checkpoint_auth=eng
    )
    assert rep["ok"] is False
    assert any("engineering" in f or "production_unlocked" in f for f in rep["failures"])


def test_d5a2_spoofed_production_unlocked_fails(monkeypatch, tmp_path):
    fix = make_sharded_ten_unit_fixture(tmp_path)
    # Locally clear pin so executor auth is not unlocked; spoofed claim must fail.
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        None,
    )
    auth = authenticate_production_checkpoint_manifest(repo_root=REPO)
    assert auth.production_unlocked is False
    spec = _spec()
    digests = fix["base_files"] | fix["unit_files"]["l19/seed42"]
    prov = {
        "model_unit": spec.model_unit,
        "family": spec.family,
        "role": spec.role,
        "seed": spec.seed,
        "base_revision": PINNED_BASE_REVISION,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "attn_implementation": "eager",
        "torch_dtype": "torch.bfloat16",
        "weight_file_sha256": digests,
        "adapter_subfolder": "l19/seed42",
        "checkpoint_trust": {"production_unlocked": True, "engineering_only": False},
        "base_weight_snapshot_dir": str(fix["base"]),
        "adapter_weight_snapshot_dir": str(fix["adapter"]),
    }
    rep = verify_unit_load_provenance(
        spec, prov, executor_checkpoint_auth=auth
    )
    assert rep["ok"] is False
    assert any(
        "spoofed" in f or "production_unlocked" in f for f in rep["failures"]
    )


def test_d5a2_incorrect_unit_evidence_fails(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "traces").mkdir(parents=True)
    (repo / "results" / "q1_implementation_prep").mkdir(parents=True)
    fix = make_sharded_ten_unit_fixture(tmp_path / "snap")
    install_temporary_production_pin(monkeypatch, repo, fix["payload"])
    auth = authenticate_production_checkpoint_manifest(repo_root=repo)
    assert auth.production_unlocked is True
    spec = _spec(seed=42)
    # Wrong subfolder / seed43 digests for seed42 unit
    wrong = unit_digests_from_auth(auth, subfolder="l19/seed43")
    prov = {
        "model_unit": spec.model_unit,
        "family": "l19",
        "role": "exploratory_dev",
        "seed": 42,
        "base_revision": PINNED_BASE_REVISION,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "attn_implementation": "eager",
        "torch_dtype": "torch.bfloat16",
        "weight_file_sha256": wrong,
        "adapter_subfolder": "l19/seed43",  # wrong for seed42
        "base_weight_snapshot_dir": str(fix["base"]),
        "adapter_weight_snapshot_dir": str(fix["adapter"]),
    }
    rep = verify_unit_load_provenance(
        spec, prov, executor_checkpoint_auth=auth
    )
    assert rep["ok"] is False
    assert any("adapter_subfolder" in f or "digest" in f for f in rep["failures"])


def test_d5a2_valid_temp_pin_passes_unit_provenance(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "src" / "traces").mkdir(parents=True)
    (repo / "results" / "q1_implementation_prep").mkdir(parents=True)
    fix = make_sharded_ten_unit_fixture(tmp_path / "snap")
    install_temporary_production_pin(monkeypatch, repo, fix["payload"])
    auth = authenticate_production_checkpoint_manifest(repo_root=repo)
    assert auth.production_unlocked is True
    assert auth.engineering_only is False
    digests = unit_digests_from_auth(auth, subfolder="l19/seed42")
    spec = _spec()
    prov = {
        "model_unit": spec.model_unit,
        "family": spec.family,
        "role": spec.role,
        "seed": spec.seed,
        "base_revision": PINNED_BASE_REVISION,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "attn_implementation": "eager",
        "torch_dtype": "torch.bfloat16",
        "weight_file_sha256": digests,
        "adapter_subfolder": "l19/seed42",
        "base_weight_snapshot_dir": str(fix["base"]),
        "adapter_weight_snapshot_dir": str(fix["adapter"]),
        "checkpoint_trust": auth.to_dict(),
    }
    rep = verify_unit_load_provenance(
        spec, prov, executor_checkpoint_auth=auth
    )
    assert rep["ok"] is True, rep["failures"]


def test_d5a2_execute_live_unit_requires_unlocked_auth(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    from traces.q1.scientific_execution import ScientificPipelineExecutor

    ex = ScientificPipelineExecutor(REPO, tmp_path / "out")
    hooks = LiveForwardHooks(
        load_model_for_unit=lambda s: {"model": object()},
        capture_donors_for_unit=lambda s, m: {},
        execute_unit=lambda s, m, d: {},
    )
    with pytest.raises(RuntimeError, match="production_unlocked|D5-A2"):
        _execute_live_unit(
            ex=ex, hooks=hooks, spec=_spec(), per_unit_audits=[], executor_checkpoint_auth=None
        )
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_f1_preserved():
    assert f1_full_set_stream_sha256() == F1_REF


def test_owner_frozen_pin_present_in_module():
    owner = "d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55"
    assert PINNED_CHECKPOINT_MANIFEST_SHA256 == owner
    import traces.q1.checkpoint_trust as ct

    assert ct.PINNED_CHECKPOINT_MANIFEST_SHA256 == owner
