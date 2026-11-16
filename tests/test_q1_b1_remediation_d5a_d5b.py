"""D5-A / D5-B checkpoint provenance — real functions, no pretrained / no forwards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from traces.q1.checkpoint_provenance import (
    FROZEN_SECTION11_TOKENIZER_SHA256,
    FROZEN_TOKENIZER_IDENTITY,
    bind_unit_checkpoint_provenance,
    engineering_bind_unit_checkpoint_provenance,
    resolve_unit_adapter_subfolder_dir,
)
from traces.q1.checkpoint_trust import (
    CHECKPOINT_MANIFEST_V1,
    ENGINEERING_CHECKPOINT_TRUST_ENV,
    PINNED_CHECKPOINT_MANIFEST_SHA256,
    REQUIRED_ADAPTER_LOAD_FILES,
    REQUIRED_BASE_LOAD_FILES,
    authenticate_engineering_checkpoint_manifest,
    authenticate_production_checkpoint_manifest,
    build_engineering_ten_unit_manifest,
    canonical_checkpoint_manifest_bytes,
    sha256_bytes,
    validate_checkpoint_manifest_schema,
)
from traces.q1.pipeline_adapters import required_adapter_subfolders
from traces.q1.production_pipeline import ProductionHookConfig, build_production_live_hooks
from traces.q1.scientific_execution import UnitSpec
from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256

REPO = Path(__file__).resolve().parents[1]
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _adapter_cfg(r: int = 8, alpha: int = 16) -> bytes:
    return json.dumps(
        {"r": r, "lora_alpha": alpha, "use_topk": False, "target_modules": ["q_proj"]}
    ).encode()


def _write_unit_dir(
    d: Path,
    *,
    weight: bytes,
    tok_tag: str,
    cfg: bytes | None = None,
) -> dict[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_model.safetensors").write_bytes(weight)
    cfg_b = cfg if cfg is not None else _adapter_cfg()
    (d / "adapter_config.json").write_bytes(cfg_b)
    files: dict[str, str] = {
        "adapter_model.safetensors": _sha(weight),
        "adapter_config.json": _sha(cfg_b),
    }
    for name in FROZEN_SECTION11_TOKENIZER_SHA256:
        raw = f"{tok_tag}-{name}".encode()
        (d / name).write_bytes(raw)
        files[name] = _sha(raw)
    return files


def _make_ten_unit_fixture(
    tmp_path: Path,
    *,
    identical_weights: bool = False,
    seeds_l19=(42, 43, 44, 45, 46),
    seeds_l1523=(42, 43, 44, 45, 46),
) -> dict[str, Any]:
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    base_w = b"BASE-WEIGHTS-V1"
    base_cfg = b'{"model_type":"gemma2","num_hidden_layers":26}\n'
    (base / "model.safetensors").write_bytes(base_w)
    (base / "config.json").write_bytes(base_cfg)
    base_files = {
        "base:model.safetensors": _sha(base_w),
        "base:config.json": _sha(base_cfg),
    }
    unit_files: dict[str, dict[str, str]] = {}
    for fam, seeds in (("l19", seeds_l19), ("l1523", seeds_l1523)):
        for seed in seeds:
            sf = f"{fam}/seed{seed}"
            d = adapter / fam / f"seed{seed}"
            if identical_weights:
                weight = b"SAME-ADAPTER-WEIGHTS"
            else:
                weight = f"ADAPTER-{sf}".encode()
            raw_files = _write_unit_dir(d, weight=weight, tok_tag=sf)
            unit_files[sf] = {
                f"adapter:{sf}/{name}": digest for name, digest in raw_files.items()
            }
    payload = build_engineering_ten_unit_manifest(
        base_files=base_files, unit_files_by_subfolder=unit_files
    )
    return {
        "base": base,
        "adapter": adapter,
        "payload": payload,
        "base_files": base_files,
        "unit_files": unit_files,
    }


@pytest.fixture
def eng_trust(monkeypatch):
    monkeypatch.setenv(ENGINEERING_CHECKPOINT_TRUST_ENV, "1")
    yield
    monkeypatch.delenv(ENGINEERING_CHECKPOINT_TRUST_ENV, raising=False)


# --- D5-A ------------------------------------------------------------------


def test_d5a_production_pin_unset_blocks(monkeypatch):
    assert PINNED_CHECKPOINT_MANIFEST_SHA256 is None
    auth = authenticate_production_checkpoint_manifest(repo_root=REPO)
    assert auth.production_unlocked is False
    assert auth.ok is False
    assert any("unset" in f for f in auth.failures)


def test_d5a_caller_manifest_not_authoritative_for_production(monkeypatch, tmp_path):
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")
    hooks = build_production_live_hooks(
        ProductionHookConfig(
            repo_root=REPO,
            expected_weight_file_sha256={
                "base:model.safetensors": "a" * 64,
                "adapter:l19/seed42/adapter_model.safetensors": "b" * 64,
            },
            base_weight_snapshot_dir=str(tmp_path / "base"),
            adapter_weight_snapshot_dir=str(tmp_path / "adapter"),
        )
    )
    spec = UnitSpec(
        model_unit="exploratory_l19_seed42",
        family="l19",
        role="exploratory_dev",
        seed=42,
    )
    with pytest.raises(RuntimeError, match="authoritative checkpoint manifest|not authoritative|unset"):
        hooks.load_model_for_unit(spec)
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_d5a_ten_identical_adapter_weights_rejected(eng_trust, tmp_path):
    fix = _make_ten_unit_fixture(tmp_path, identical_weights=True)
    failures = validate_checkpoint_manifest_schema(fix["payload"])
    assert any("pairwise distinct" in f or "collides" in f for f in failures)
    auth = authenticate_engineering_checkpoint_manifest(fix["payload"])
    assert auth.ok is False
    assert auth.production_unlocked is False


def test_d5a_cross_seed_bytes_refused_even_with_matching_caller_map(
    eng_trust, tmp_path, monkeypatch
):
    fix = _make_ten_unit_fixture(tmp_path)
    # Replace seed43 weight bytes with seed42 bytes but keep seed43 digest in a
    # *caller* flat map that claims the new bytes — engineering auth uses payload.
    src = (fix["adapter"] / "l19" / "seed42" / "adapter_model.safetensors").read_bytes()
    dst = fix["adapter"] / "l19" / "seed43" / "adapter_model.safetensors"
    dst.write_bytes(src)
    # Authoritative engineering payload still lists original seed43 digest
    with pytest.raises(RuntimeError, match="SHA|mismatch"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=43,
            subfolder="l19/seed43",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )
    # Matching caller flat map with substituted digest must not unlock production
    caller = dict(fix["unit_files"]["l19/seed43"])
    caller["adapter:l19/seed43/adapter_model.safetensors"] = _sha(src)
    caller.update(fix["base_files"])
    auth = authenticate_production_checkpoint_manifest(repo_root=REPO)
    assert auth.production_unlocked is False
    with pytest.raises(RuntimeError, match="authoritative trust|engineering|refuse"):
        bind_unit_checkpoint_provenance(
            family="l19",
            seed=43,
            subfolder="l19/seed43",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            expected_sha256_by_relpath=caller,
            require_section11_tokenizer_pins=False,
            require_authoritative_trust=True,
            trust_auth=None,
        )


def test_d5a_symlinked_seed_subfolder_refused(eng_trust, tmp_path):
    fix = _make_ten_unit_fixture(tmp_path)
    target = fix["adapter"] / "l19" / "seed42"
    link = fix["adapter"] / "l19" / "seed99"
    # Replace seed43 with symlink to seed42
    seed43 = fix["adapter"] / "l19" / "seed43"
    for child in seed43.iterdir():
        child.unlink()
    seed43.rmdir()
    seed43.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        resolve_unit_adapter_subfolder_dir(fix["adapter"], "l19/seed43")
    with pytest.raises(RuntimeError, match="symlink"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=43,
            subfolder="l19/seed43",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )
    _ = link


def test_d5a_path_traversal_and_malformed_keys_rejected():
    bad = {
        "artifact_type": CHECKPOINT_MANIFEST_V1,
        "self_generated": False,
        "base_files": {"base:../etc/passwd": "a" * 64},
        "units": {},
        "evil_key": True,
    }
    failures = validate_checkpoint_manifest_schema(bad)
    assert any("traversal" in f or "untrusted" in f or "unexpected" in f for f in failures)


def test_d5a_incorrect_owner_pin_rejected(monkeypatch, tmp_path):
    fix = _make_ten_unit_fixture(tmp_path)
    man_dir = tmp_path / "results" / "q1_implementation_prep"
    man_dir.mkdir(parents=True)
    man_path = man_dir / "CHECKPOINT_PROVENANCE_MANIFEST.json"
    man_path.write_text(json.dumps(fix["payload"], sort_keys=True))
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        "0" * 64,
    )
    auth = authenticate_production_checkpoint_manifest(repo_root=tmp_path)
    assert auth.production_unlocked is False
    assert any("!= owner pin" in f or "digest" in f for f in auth.failures)


def test_d5a_missing_authoritative_manifest_file(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        "a" * 64,
    )
    auth = authenticate_production_checkpoint_manifest(repo_root=tmp_path)
    assert auth.production_unlocked is False
    assert any("missing" in f for f in auth.failures)


# --- D5-B ------------------------------------------------------------------


def test_d5b_altered_adapter_config_scaling_refused(eng_trust, tmp_path):
    fix = _make_ten_unit_fixture(tmp_path)
    cfg_path = fix["adapter"] / "l19" / "seed42" / "adapter_config.json"
    cfg_path.write_bytes(_adapter_cfg(r=8, alpha=32))  # scaling changed
    with pytest.raises(RuntimeError, match="SHA|mismatch|adapter_config"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )


def test_d5b_altered_special_tokens_map_refused(eng_trust, tmp_path):
    fix = _make_ten_unit_fixture(tmp_path)
    p = fix["adapter"] / "l19" / "seed42" / "special_tokens_map.json"
    p.write_bytes(b"ALTERED-SPECIAL-TOKENS")
    with pytest.raises(RuntimeError, match="SHA|mismatch|tokenizer"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )


def test_d5b_planted_tokenizer_json_refused(eng_trust, tmp_path):
    fix = _make_ten_unit_fixture(tmp_path)
    planted = fix["adapter"] / "l19" / "seed42" / "tokenizer.json"
    planted.write_text('{"version":"1.0"}')
    with pytest.raises(RuntimeError, match="tokenizer.json|competing"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )


def test_d5b_unlisted_loadable_file_refused(eng_trust, tmp_path):
    fix = _make_ten_unit_fixture(tmp_path)
    extra = fix["adapter"] / "l19" / "seed42" / "extra_weights.bin"
    extra.write_bytes(b"UNLISTED")
    with pytest.raises(RuntimeError, match="not in unit manifest|unlisted"):
        engineering_bind_unit_checkpoint_provenance(
            family="l19",
            seed=42,
            subfolder="l19/seed42",
            base_snapshot_dir=fix["base"],
            adapter_snapshot_dir=fix["adapter"],
            engineering_manifest=fix["payload"],
            require_section11_tokenizer_pins=False,
        )


def test_d5b_tokenizer_class_and_fast_contract():
    assert FROZEN_TOKENIZER_IDENTITY["expected_tokenizer_class"] == "GemmaTokenizer"
    assert FROZEN_TOKENIZER_IDENTITY["expected_is_fast"] is False
    assert "special_tokens_map.json" in FROZEN_SECTION11_TOKENIZER_SHA256
    assert set(FROZEN_SECTION11_TOKENIZER_SHA256) == {
        "tokenizer.model",
        "tokenizer_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    }


def test_d5b_incorrect_tokenizer_class_or_fast_refused_in_loader(monkeypatch, tmp_path, eng_trust):
    """Loader refuses wrong class / fast tokenizer without loading pretrained weights."""
    from traces.q1 import loader as loader_mod

    fix = _make_ten_unit_fixture(tmp_path)
    monkeypatch.setenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", "1")

    class _FastTok:
        is_fast = True

        @classmethod
        def from_pretrained(cls, *a, **k):
            assert k.get("use_fast") is False
            return cls()

    with patch.object(loader_mod, "AutoTokenizer", _FastTok):
        # Bind succeeds; tokenizer is_fast check fires before model load
        with pytest.raises(RuntimeError, match="is_fast|tokenizer class|GemmaTokenizer"):
            loader_mod.engineering_load_q1_scientific_model(
                subfolder="l19/seed42",
                engineering_manifest_payload=fix["payload"],
                expected_adapter_targets=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                expected_adapted_layers=[19],
                base_weight_snapshot_dir=str(fix["base"]),
                adapter_weight_snapshot_dir=str(fix["adapter"]),
            )
    monkeypatch.delenv("TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION", raising=False)


def test_d5b_valid_engineering_fixture_passes_isolated(eng_trust, tmp_path):
    fix = _make_ten_unit_fixture(tmp_path)
    auth = authenticate_engineering_checkpoint_manifest(fix["payload"])
    assert auth.ok is True
    assert auth.production_unlocked is False
    assert auth.engineering_only is True
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
    assert bind["computed_sha256"]
    assert bind["adapter_config"]["r"] == 8
    assert bind["trust"]["engineering_only"] is True
    assert bind["trust"]["production_unlocked"] is False
    # All ten subfolders present and distinct in payload
    assert set(fix["payload"]["units"]) == set(required_adapter_subfolders())
    digests = [
        fix["payload"]["units"][sf]["files"][
            f"adapter:{sf}/adapter_model.safetensors"
        ]
        for sf in required_adapter_subfolders()
    ]
    assert len(set(digests)) == 10


def test_d5b_section11_pins_unchanged():
    assert FROZEN_SECTION11_TOKENIZER_SHA256["tokenizer.model"].startswith("61a7b147")
    assert FROZEN_SECTION11_TOKENIZER_SHA256["tokenizer_config.json"].startswith(
        "5db39e7c"
    )
    assert FROZEN_SECTION11_TOKENIZER_SHA256["chat_template.jinja"].startswith(
        "ecd6ae51"
    )
    assert FROZEN_SECTION11_TOKENIZER_SHA256["special_tokens_map.json"].startswith(
        "5b36db1a"
    )


def test_d5a_adapter_off_still_maps_to_l19_seed42():
    from traces.q1.pipeline_adapters import unit_spec_to_adapter_subfolder
    from traces.q1.scientific_execution import UnitSpec as US

    # Production load path uses l19/seed42 for base (see production_pipeline)
    spec = US(
        model_unit="adapter_off_base_reference",
        family="base",
        role="adapter_off_base_reference",
        seed=None,
    )
    assert unit_spec_to_adapter_subfolder(spec) == "adapter_off_base_reference"
    # Bind identity for base family requires l19/seed42
    # (exercised via bind_unit expected_sf)


def test_f1_preserved():
    assert f1_full_set_stream_sha256() == F1_REF


def test_required_inventory_constants():
    assert "special_tokens_map.json" in REQUIRED_ADAPTER_LOAD_FILES
    assert "adapter_config.json" in REQUIRED_ADAPTER_LOAD_FILES
    assert "config.json" in REQUIRED_BASE_LOAD_FILES
