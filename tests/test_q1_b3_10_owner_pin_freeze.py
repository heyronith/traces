"""B3-10 post-freeze CPU-only checkpoint-trust verification (no model load)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from traces.q1.checkpoint_trust import (
    PINNED_CHECKPOINT_MANIFEST_SHA256,
    PRODUCTION_CHECKPOINT_MANIFEST_RELPATH,
    authenticate_production_checkpoint_manifest,
    canonical_checkpoint_manifest_bytes,
    sha256_bytes,
    validate_checkpoint_manifest_schema,
)

REPO = Path(__file__).resolve().parents[1]
OWNER_PIN = "d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55"
CANDIDATE = (
    REPO
    / "results/q1_implementation_prep/B3_CANDIDATE_CHECKPOINT_MANIFEST.json"
)
PRODUCTION = REPO / PRODUCTION_CHECKPOINT_MANIFEST_RELPATH
GIT_BLOB = "7887be2f8d584dd603b871a8151b1e4d9df947db"
F1_REF = "fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed"


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def test_owner_pin_equals_audited_canonical_digest():
    assert PINNED_CHECKPOINT_MANIFEST_SHA256 == OWNER_PIN


def test_production_manifest_bytes_match_candidate_and_git_blob():
    prod = PRODUCTION.read_bytes()
    cand = CANDIDATE.read_bytes()
    assert prod == cand
    assert _git_blob_sha1(prod) == GIT_BLOB
    assert _git_blob_sha1(cand) == GIT_BLOB


def test_canonical_sha256_matches_owner_pin():
    payload = json.loads(PRODUCTION.read_text(encoding="utf-8"))
    assert validate_checkpoint_manifest_schema(payload) == []
    digest = sha256_bytes(canonical_checkpoint_manifest_bytes(payload))
    assert digest == OWNER_PIN
    # Historical candidate notes preserved verbatim (changing them changes the hash).
    notes = " ".join(payload.get("notes") or [])
    assert "B3_CANDIDATE_ONLY" in notes
    assert "must remain unset until owner freeze" in notes


def test_authenticate_production_manifest_unlocks_identity_only():
    auth = authenticate_production_checkpoint_manifest(repo_root=REPO)
    assert auth.ok is True
    assert auth.production_unlocked is True
    assert auth.engineering_only is False
    assert auth.failures == []
    assert auth.computed_manifest_sha256 == OWNER_PIN
    assert auth.mode == "pinned_digest"
    # Manifest identity unlock ≠ scientific authorization.
    assert auth.to_dict()["production_unlocked"] is True


def test_missing_manifest_blocks(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        OWNER_PIN,
    )
    auth = authenticate_production_checkpoint_manifest(repo_root=tmp_path)
    assert auth.ok is False
    assert auth.production_unlocked is False
    assert any("missing" in f for f in auth.failures)


def test_altered_manifest_bytes_block(tmp_path, monkeypatch):
    man_dir = tmp_path / "results" / "q1_implementation_prep"
    man_dir.mkdir(parents=True)
    payload = json.loads(PRODUCTION.read_text(encoding="utf-8"))
    payload = dict(payload)
    payload["notes"] = list(payload.get("notes") or []) + ["tampered"]
    (man_dir / "CHECKPOINT_PROVENANCE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        OWNER_PIN,
    )
    auth = authenticate_production_checkpoint_manifest(repo_root=tmp_path)
    assert auth.ok is False
    assert auth.production_unlocked is False
    assert any("!= owner pin" in f or "digest" in f for f in auth.failures)


def test_wrong_owner_pin_blocks(monkeypatch):
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        "0" * 64,
    )
    auth = authenticate_production_checkpoint_manifest(repo_root=REPO)
    assert auth.ok is False
    assert auth.production_unlocked is False
    assert any("!= owner pin" in f or "digest" in f for f in auth.failures)


def test_unpinned_default_blocks(monkeypatch):
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        None,
    )
    auth = authenticate_production_checkpoint_manifest(repo_root=REPO)
    assert auth.ok is False
    assert auth.production_unlocked is False
    assert auth.mode == "pin_unset"
    assert any("unset" in f for f in auth.failures)


def test_caller_supplied_forged_hash_not_used_by_production_auth(tmp_path, monkeypatch):
    """Production auth ignores caller digests; only fixed-path + owner pin."""
    man_dir = tmp_path / "results" / "q1_implementation_prep"
    man_dir.mkdir(parents=True)
    # Write authentic production bytes under a temp root.
    man_dir.joinpath("CHECKPOINT_PROVENANCE_MANIFEST.json").write_bytes(
        PRODUCTION.read_bytes()
    )
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        OWNER_PIN,
    )
    auth = authenticate_production_checkpoint_manifest(repo_root=tmp_path)
    assert auth.ok is True
    assert auth.computed_manifest_sha256 == OWNER_PIN
    # Forged alternate pin does not authenticate even if caller wishes otherwise.
    monkeypatch.setattr(
        "traces.q1.checkpoint_trust.PINNED_CHECKPOINT_MANIFEST_SHA256",
        "f" * 64,
    )
    auth_forged = authenticate_production_checkpoint_manifest(repo_root=tmp_path)
    assert auth_forged.ok is False
    assert auth_forged.production_unlocked is False


def test_f1_unchanged_after_pin_freeze():
    from traces.q1.stats_f1_f2 import f1_full_set_stream_sha256

    assert f1_full_set_stream_sha256() == F1_REF
