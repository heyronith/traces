"""Shared synthetic checkpoint fixtures for D5 integration tests (no pretrained)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from traces.q1.checkpoint_provenance import FROZEN_SECTION11_TOKENIZER_SHA256
from traces.q1.checkpoint_trust import (
    CHECKPOINT_MANIFEST_V1,
    PRODUCTION_CHECKPOINT_MANIFEST_RELPATH,
    build_engineering_ten_unit_manifest,
    canonical_checkpoint_manifest_bytes,
    sha256_bytes,
)
from traces.q1.pipeline_adapters import required_adapter_subfolders


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_sharded_base(
    base: Path,
    *,
    n_shards: int = 3,
    shard_payload_prefix: bytes = b"SHARD",
) -> dict[str, str]:
    """Write config + index + N shards; return role-tagged digests."""
    base.mkdir(parents=True, exist_ok=True)
    cfg = b'{"model_type":"gemma2","num_hidden_layers":26}\n'
    (base / "config.json").write_bytes(cfg)
    shards = [f"model-{i:05d}-of-{n_shards:05d}.safetensors" for i in range(1, n_shards + 1)]
    weight_map = {f"layer.{i}.weight": shards[i % n_shards] for i in range(6)}
    for i, name in enumerate(shards):
        (base / name).write_bytes(shard_payload_prefix + f"-{i+1}".encode())
    index = {"metadata": {"total_size": 1}, "weight_map": weight_map}
    index_bytes = json.dumps(index, sort_keys=True).encode()
    (base / "model.safetensors.index.json").write_bytes(index_bytes)
    files = {
        "base:config.json": _sha(cfg),
        "base:model.safetensors.index.json": _sha(index_bytes),
    }
    for name in shards:
        files[f"base:{name}"] = _sha((base / name).read_bytes())
    return files


def _adapter_cfg(r: int = 8, alpha: int = 16) -> bytes:
    return json.dumps(
        {"r": r, "lora_alpha": alpha, "use_topk": False, "target_modules": ["q_proj"]}
    ).encode()


def write_ten_unit_adapter_tree(
    adapter: Path,
    *,
    identical_weights: bool = False,
) -> dict[str, dict[str, str]]:
    adapter.mkdir(parents=True, exist_ok=True)
    unit_files: dict[str, dict[str, str]] = {}
    for sf in required_adapter_subfolders():
        fam, seed_tok = sf.split("/", 1)
        d = adapter / fam / seed_tok
        d.mkdir(parents=True, exist_ok=True)
        weight = b"SAME-ADAPTER-WEIGHTS" if identical_weights else f"ADAPTER-{sf}".encode()
        (d / "adapter_model.safetensors").write_bytes(weight)
        cfg = _adapter_cfg()
        (d / "adapter_config.json").write_bytes(cfg)
        files = {
            f"adapter:{sf}/adapter_model.safetensors": _sha(weight),
            f"adapter:{sf}/adapter_config.json": _sha(cfg),
        }
        for name in FROZEN_SECTION11_TOKENIZER_SHA256:
            raw = f"{sf}-{name}".encode()
            (d / name).write_bytes(raw)
            files[f"adapter:{sf}/{name}"] = _sha(raw)
        unit_files[sf] = files
    return unit_files


def make_sharded_ten_unit_fixture(tmp_path: Path, **kw: Any) -> dict[str, Any]:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base_files = write_sharded_base(base)
    unit_files = write_ten_unit_adapter_tree(
        adapter, identical_weights=bool(kw.get("identical_weights", False))
    )
    payload = build_engineering_ten_unit_manifest(
        base_files=base_files, unit_files_by_subfolder=unit_files
    )
    return {
        "base": base,
        "adapter": adapter,
        "base_files": base_files,
        "unit_files": unit_files,
        "payload": payload,
    }


def stub_live_pipeline_checkpoint_trust(monkeypatch: Any) -> None:
    """For non-D5 live-pipeline tests: unlock executor trust + skip byte bind.

    Does **not** set the real owner pin. Isolates D5-A2 so S2/B1 mock-hook
    suites can exercise other fail-closed gates.
    """
    from unittest.mock import MagicMock

    auth = MagicMock(
        production_unlocked=True,
        engineering_only=False,
        ok=True,
        failures=[],
        payload={"units": {}},
        mode="test_stub_non_d5",
    )
    auth.to_dict = lambda: {
        "ok": True,
        "production_unlocked": True,
        "engineering_only": False,
        "mode": "test_stub_non_d5",
    }
    # Patch the trust module (live path imports inside the function body).
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.authenticate_production_checkpoint_manifest",
        lambda **kwargs: auth,
    )


def install_temporary_production_pin(
    monkeypatch: Any,
    repo_root: Path,
    payload: Mapping[str, Any],
) -> str:
    """Write fixed-path manifest under repo_root and pin its digest (tests only)."""
    body = dict(payload)
    body.setdefault("artifact_type", CHECKPOINT_MANIFEST_V1)
    body.pop("self_generated", None)
    man_path = Path(repo_root) / PRODUCTION_CHECKPOINT_MANIFEST_RELPATH
    man_path.parent.mkdir(parents=True, exist_ok=True)
    canon = canonical_checkpoint_manifest_bytes(body)
    digest = sha256_bytes(canon)
    # Store pretty JSON; auth uses canonical bytes of parsed object
    man_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    # Recompute from file parse path used by auth
    loaded = json.loads(man_path.read_text())
    digest = sha256_bytes(canonical_checkpoint_manifest_bytes(loaded))
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        digest,
    )
    return digest
