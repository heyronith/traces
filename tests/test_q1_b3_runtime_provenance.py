"""B3 checkpoint + runtime-lock engineering — synthetic fail-closed tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from traces.q1.checkpoint_provenance import (
    assert_load_uses_verified_snapshot_bytes,
    assess_checkpoint_manifest,
    require_verified_snapshot_for_load,
    verify_snapshot_bytes_match_manifest,
)
from traces.q1.pipeline_adapters import required_adapter_subfolders
from traces.q1.runtime_lock import claim_production_lock_exists
from traces.q1.runtime_spec import REQUIRED_ATTN_IMPLEMENTATION, REQUIRED_TORCH_DTYPE


def test_b3_required_adapter_subfolders_complete():
    subs = required_adapter_subfolders()
    assert len(subs) == 10
    assert "l19/seed45" in subs
    assert "l1523/seed42" in subs


def test_b3_manifest_blocked_without_authoritative_hashes():
    status = assess_checkpoint_manifest()
    assert status.blocked is True
    assert status.ok is False
    assert any("unavailable" in f or "not present" in f for f in status.failures)


def test_b3_refuse_invented_production_lock_claim():
    import json

    # Synthetic fixture must never count as production
    assert claim_production_lock_exists(None) is False
    synth = Path("results/q1_implementation_prep/runtime_lock_synthetic_fixture.json")
    if synth.is_file():
        doc = json.loads(synth.read_text())
        assert claim_production_lock_exists(doc) is False
    # Flag alone is insufficient
    assert (
        claim_production_lock_exists(
            {"production_image_lock": True, "source": "built_image"}
        )
        is False
    )


def test_b3_snapshot_byte_verify_and_mismatch(tmp_path: Path):
    payload = b"not-a-real-checkpoint"
    digest = hashlib.sha256(payload).hexdigest()
    f = tmp_path / "model.safetensors"
    f.write_bytes(payload)
    ok = verify_snapshot_bytes_match_manifest(
        snapshot_dir=tmp_path,
        expected_sha256_by_relpath={"model.safetensors": digest},
    )
    assert ok["ok"] is True
    bad = verify_snapshot_bytes_match_manifest(
        snapshot_dir=tmp_path,
        expected_sha256_by_relpath={"model.safetensors": "0" * 64},
    )
    assert bad["ok"] is False


def test_b3_assert_load_uses_verified_bytes(tmp_path: Path):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    (base / "model.safetensors").write_bytes(b"base-bytes")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-bytes")
    expected = {
        "base:model.safetensors": hashlib.sha256(b"base-bytes").hexdigest(),
        "l19/seed45/adapter_model.safetensors": hashlib.sha256(
            b"adapter-bytes"
        ).hexdigest(),
    }
    # Relative path under adapter dir must exist as written
    (adapter / "l19").mkdir()
    (adapter / "l19" / "seed45").mkdir()
    (adapter / "l19" / "seed45" / "adapter_model.safetensors").write_bytes(
        b"adapter-bytes"
    )
    assert_load_uses_verified_snapshot_bytes(
        base_snapshot_dir=base,
        adapter_snapshot_dir=adapter,
        expected_sha256_by_relpath=expected,
    )
    with pytest.raises(RuntimeError, match="verification failed|mismatch"):
        assert_load_uses_verified_snapshot_bytes(
            base_snapshot_dir=base,
            adapter_snapshot_dir=adapter,
            expected_sha256_by_relpath={
                "base:model.safetensors": "a" * 64,
                "l19/seed45/adapter_model.safetensors": "b" * 64,
            },
        )


def test_b3_require_verified_snapshot_fail_closed():
    out = require_verified_snapshot_for_load(manifest_path=None, snapshot_root=None)
    assert out["ok"] is False
    assert out["blocked"] is True


def test_b3_bf16_eager_enforcement_constants():
    import torch

    assert REQUIRED_TORCH_DTYPE == torch.bfloat16
    assert REQUIRED_ATTN_IMPLEMENTATION == "eager"
