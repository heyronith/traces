"""Synthetic CPU tests for B3 Modal byte-verify helpers — no downloads / no GPU."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from traces.q1.b3_modal_byte_verify import (
    BILLABLE_CONFIRM_ENV,
    OWNER_APPROVED_CAP_USD,
    OWNER_BUDGET_ENV,
    B3VerifyError,
    assert_adapter_weights_pairwise_distinct,
    assert_regular_file_no_symlink,
    build_candidate_manifest,
    estimate_modal_cpu_job_usd,
    expected_ten_units,
    inventory_materialized_snapshot,
    parse_owner_budget_usd,
    preflight_local,
    reject_unsafe_relpath,
    require_billable_gates,
    sha256_file_chunked,
    verify_adapter_unit_inventory,
)
from traces.q1.checkpoint_provenance import FROZEN_SECTION11_TOKENIZER_SHA256
from traces.q1.checkpoint_trust import (
    PINNED_CHECKPOINT_MANIFEST_SHA256,
    REQUIRED_ADAPTER_LOAD_FILES,
    canonical_checkpoint_manifest_bytes,
    sha256_bytes,
    validate_checkpoint_manifest_schema,
)
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256

F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_sharded_base(base: Path) -> dict[str, str]:
    base.mkdir(parents=True, exist_ok=True)
    shard_a = b"SHARD-A-BYTES"
    shard_b = b"SHARD-B-BYTES"
    (base / "model-00001-of-00002.safetensors").write_bytes(shard_a)
    (base / "model-00002-of-00002.safetensors").write_bytes(shard_b)
    index = {
        "metadata": {"total_size": len(shard_a) + len(shard_b)},
        "weight_map": {
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "model.layers.1.weight": "model-00002-of-00002.safetensors",
        },
    }
    index_b = json.dumps(index, sort_keys=True).encode()
    (base / "model.safetensors.index.json").write_bytes(index_b)
    cfg = b'{"architectures":["Gemma2ForCausalLM"]}'
    (base / "config.json").write_bytes(cfg)
    return {
        "base:config.json": _sha(cfg),
        "base:model.safetensors.index.json": _sha(index_b),
        "base:model-00001-of-00002.safetensors": _sha(shard_a),
        "base:model-00002-of-00002.safetensors": _sha(shard_b),
    }


def _write_unit(d: Path, *, weight: bytes, tok_tag: str) -> dict[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(weight)
    cfg = json.dumps({"r": 8, "lora_alpha": 16}).encode()
    (d / "adapter_config.json").write_bytes(cfg)
    out = {
        "adapter_model.safetensors": _sha(weight),
        "adapter_config.json": _sha(cfg),
    }
    for name in FROZEN_SECTION11_TOKENIZER_SHA256:
        # Synthetic tokens intentionally do NOT match frozen pins unless asked.
        raw = f"{tok_tag}-{name}".encode()
        (d / name).write_bytes(raw)
        out[name] = _sha(raw)
    return out


def _write_unit_matching_pins(d: Path, *, weight: bytes) -> dict[str, str]:
    """Write §11 files whose SHA match frozen pins (tiny synthetic content).

    Frozen pins are fixed digests; we cannot invent content for them. Tests that
    need schema-only candidate builds use non-matching synthetic tokens and
    call build_candidate_manifest directly with digests. Inventory path that
    enforces §11 will BLOCK — covered separately.
    """
    return _write_unit(d, weight=weight, tok_tag="synth")


def test_pin_remains_unset():
    assert PINNED_CHECKPOINT_MANIFEST_SHA256 is None


def test_ten_units_not_five():
    units = expected_ten_units()
    assert len(units) == 10
    assert "l19/seed42" in units and "l1523/seed46" in units
    assert units.count("l19/seed42") == 1


def test_budget_missing_fails():
    with pytest.raises(B3VerifyError, match="missing"):
        parse_owner_budget_usd(None)
    with pytest.raises(B3VerifyError, match="missing"):
        parse_owner_budget_usd("")


def test_budget_zero_fails():
    with pytest.raises(B3VerifyError, match="> 0"):
        parse_owner_budget_usd("0")


def test_budget_invalid_fails():
    with pytest.raises(B3VerifyError, match="not a valid"):
        parse_owner_budget_usd("abc")


def test_billable_gates_require_chat_confirm_and_exact_10(monkeypatch):
    monkeypatch.delenv(BILLABLE_CONFIRM_ENV, raising=False)
    monkeypatch.setenv(OWNER_BUDGET_ENV, "10")
    with pytest.raises(B3VerifyError, match="B3_CHAT_CAP_CONFIRMED"):
        require_billable_gates()
    monkeypatch.setenv(BILLABLE_CONFIRM_ENV, "1")
    monkeypatch.setenv(OWNER_BUDGET_ENV, "9")
    with pytest.raises(B3VerifyError, match="must equal"):
        require_billable_gates()
    monkeypatch.setenv(OWNER_BUDGET_ENV, "10")
    gates = require_billable_gates()
    assert gates["owner_approved_max_usd"] == OWNER_APPROVED_CAP_USD
    assert gates["cost_estimate"]["safety_bound_usd"] <= 10


def test_cost_estimate_under_cap():
    est = estimate_modal_cpu_job_usd()
    assert est["safety_bound_usd"] < OWNER_APPROVED_CAP_USD


def test_path_traversal_rejected():
    with pytest.raises(B3VerifyError, match="traversal"):
        reject_unsafe_relpath("../etc/passwd")
    with pytest.raises(B3VerifyError, match="absolute"):
        reject_unsafe_relpath("/tmp/x")


def test_symlink_refused(tmp_path: Path):
    real = tmp_path / "real.bin"
    real.write_bytes(b"abc")
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    with pytest.raises(B3VerifyError, match="symlink"):
        assert_regular_file_no_symlink(link)


def test_chunked_hash(tmp_path: Path):
    p = tmp_path / "f.bin"
    data = b"x" * 5000
    p.write_bytes(data)
    digest, size = sha256_file_chunked(p, chunk_size=128)
    assert size == 5000
    assert digest == _sha(data)


def test_shuffled_identity_and_duplicate_weight(tmp_path: Path):
    adapter = tmp_path / "adapters"
    digests = {}
    for sf in expected_ten_units():
        d = adapter.joinpath(*sf.split("/"))
        _write_unit(d, weight=b"SAME", tok_tag=sf)
        digests[sf] = _sha(b"SAME")
    with pytest.raises(B3VerifyError, match="pairwise distinct"):
        assert_adapter_weights_pairwise_distinct(digests)


def test_forbidden_tokenizer_blocks(tmp_path: Path):
    d = tmp_path / "l19" / "seed42"
    _write_unit(d, weight=b"W42", tok_tag="u")
    (d / "tokenizer.json").write_text("{}")
    inv = verify_adapter_unit_inventory(d, subfolder="l19/seed42")
    assert inv["ok"] is False
    assert any("forbidden" in f for f in inv["failures"])


def test_extra_file_blocks(tmp_path: Path):
    d = tmp_path / "l19" / "seed42"
    _write_unit(d, weight=b"W42", tok_tag="u")
    (d / "topk_config.json").write_text("{}")
    inv = verify_adapter_unit_inventory(d, subfolder="l19/seed42")
    assert inv["ok"] is False
    assert any("unexpected extra" in f for f in inv["failures"])


def test_wrong_shard_index_blocks(tmp_path: Path):
    base = tmp_path / "base"
    _write_sharded_base(base)
    # Corrupt index to reference missing shard
    bad = {
        "weight_map": {"t": "missing-shard.safetensors"},
        "metadata": {},
    }
    (base / "model.safetensors.index.json").write_text(json.dumps(bad))
    adapter = tmp_path / "adapters"
    for sf in expected_ten_units():
        _write_unit(adapter.joinpath(*sf.split("/")), weight=f"W-{sf}".encode(), tok_tag=sf)
    report, cand = inventory_materialized_snapshot(base_dir=base, adapter_root=adapter)
    assert report.ok is False
    assert cand is None


def test_candidate_manifest_schema_and_not_production_path(tmp_path: Path):
    base_digests = _write_sharded_base(tmp_path / "base")
    unit_digests = {}
    for sf in expected_ten_units():
        files = _write_unit(
            tmp_path / "adapters" / Path(sf),
            weight=f"W-{sf}".encode(),
            tok_tag=sf,
        )
        unit_digests[sf] = files
    payload = build_candidate_manifest(
        base_digests=base_digests, unit_digests=unit_digests
    )
    assert validate_checkpoint_manifest_schema(payload) == []
    assert payload["artifact_type"].startswith("q1_checkpoint")
    assert "B3_CANDIDATE_ONLY" in " ".join(payload["notes"])
    # Canonical digest stable
    d1 = sha256_bytes(canonical_checkpoint_manifest_bytes(payload))
    d2 = sha256_bytes(canonical_checkpoint_manifest_bytes(payload))
    assert d1 == d2 and len(d1) == 64
    # Must never look like production path write
    assert not (tmp_path / "CHECKPOINT_PROVENANCE_MANIFEST.json").exists()


def test_section11_mismatch_blocks_inventory(tmp_path: Path):
    base = tmp_path / "base"
    _write_sharded_base(base)
    adapter = tmp_path / "adapters"
    for sf in expected_ten_units():
        _write_unit(adapter.joinpath(*sf.split("/")), weight=f"W-{sf}".encode(), tok_tag=sf)
    report, cand = inventory_materialized_snapshot(base_dir=base, adapter_root=adapter)
    assert report.ok is False
    assert cand is None
    assert any("§11 mismatch" in f or "section11" in f.lower() or "mismatch" in f for f in report.failures)


def test_preflight_no_download(monkeypatch):
    monkeypatch.delenv(OWNER_BUDGET_ENV, raising=False)
    monkeypatch.delenv(BILLABLE_CONFIRM_ENV, raising=False)
    out = preflight_local()
    assert out["ok"] is True
    assert out["billable_download_permitted"] is False
    assert out["production_pin"] is None
    assert len(out["ten_units"]) == 10
    json.dumps(out, allow_nan=False)


def test_f1_unchanged():
    assert f1_full_set_stream_sha256() == F1_REF


def test_required_adapter_file_count():
    assert set(REQUIRED_ADAPTER_LOAD_FILES) == {
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    }
