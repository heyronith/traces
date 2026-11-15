"""One-way empirical provenance trust (S2-1).

Empirical science artifacts must **not** embed the provenance manifest hash.
The canonical execution manifest lists artifact SHA256 digests. Production
authentication obtains the manifest from a fixed/pinned source and verifies
either:

1. A detached signature against a separately trusted owner key, or
2. An exact manifest digest pinned in an independently reviewed commit constant.

Exact approval text alone is never authorization. A non-empty
``independent_approval_record`` field is ignored (not accepted as proof).

Test-only trust anchors may exercise the cryptographic verifier but **must
never** unlock production scientific execution.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

# Production pin — empty until an independently reviewed commit sets a digest.
# Never invent a real owner key, approval, signature, or manifest.
PINNED_EMPIRICAL_MANIFEST_SHA256: str | None = None

# Fixed relative path for a future owner-approved manifest (not inventing contents).
PRODUCTION_MANIFEST_RELPATH = (
    "results/q1_implementation_prep/EMPIRICAL_EXECUTION_MANIFEST.json"
)
PRODUCTION_OWNER_KEY_RELPATH = (
    "results/q1_implementation_prep/EMPIRICAL_OWNER_PUBLIC_KEY.hmac"
)
PRODUCTION_SIGNATURE_RELPATH = (
    "results/q1_implementation_prep/EMPIRICAL_EXECUTION_MANIFEST.sig"
)

ARTIFACT_PROVENANCE_MANIFEST_V1 = "q1_empirical_execution_provenance_manifest_v1"
TEST_TRUST_ENV = "TRACES_Q1_EMPIRICAL_TEST_TRUST"

# Required artifact type keys the manifest must digest-bind
REQUIRED_MANIFEST_ARTIFACT_TYPES = frozenset(
    {
        "q1_empirical_identity_suite_v1",
        "q1_empirical_f4_tolerance_table_v1",
        "q1_empirical_double_bos_64_pair_scan_v1",
    }
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path | str) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    """Deterministic encoding for signing/digest (excludes any signature field)."""
    body = {k: v for k, v in payload.items() if k not in {"signature", "detached_signature"}}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest_hmac(manifest_bytes: bytes, key: bytes) -> str:
    """Detached HMAC-SHA256 hex digest (stdlib; no third-party crypto dep)."""
    return hmac.new(key, manifest_bytes, hashlib.sha256).hexdigest()


def verify_detached_hmac(
    manifest_bytes: bytes, signature_hex: str, trusted_key: bytes
) -> bool:
    if not signature_hex or not trusted_key:
        return False
    expected = sign_manifest_hmac(manifest_bytes, trusted_key)
    try:
        return hmac.compare_digest(expected.lower(), str(signature_hex).lower())
    except (TypeError, ValueError):
        return False


@dataclass
class CryptoVerifyResult:
    ok: bool
    mode: str  # "pinned_digest" | "detached_hmac" | "test_hmac" | "none"
    failures: list[str] = field(default_factory=list)
    computed_manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "failures": list(self.failures),
            "computed_manifest_sha256": self.computed_manifest_sha256,
        }


def verify_manifest_cryptography(
    *,
    manifest_bytes: bytes,
    signature_hex: str | None = None,
    trusted_key: bytes | None = None,
    pinned_digest: str | None = PINNED_EMPIRICAL_MANIFEST_SHA256,
    allow_test_trust: bool = False,
) -> CryptoVerifyResult:
    """Pure cryptographic / pin verification (no production unlock semantics).

    When ``allow_test_trust=True`` and ``TRACES_Q1_EMPIRICAL_TEST_TRUST=1``,
    a test key may validate the signature for unit tests of the verifier.
    """
    computed = sha256_bytes(manifest_bytes)
    failures: list[str] = []

    if pinned_digest:
        if computed.lower() == str(pinned_digest).lower():
            return CryptoVerifyResult(
                ok=True, mode="pinned_digest", computed_manifest_sha256=computed
            )
        failures.append(
            f"manifest digest {computed} != pinned {pinned_digest}"
        )

    if signature_hex and trusted_key:
        if verify_detached_hmac(manifest_bytes, signature_hex, trusted_key):
            mode = "test_hmac" if allow_test_trust else "detached_hmac"
            return CryptoVerifyResult(
                ok=True, mode=mode, computed_manifest_sha256=computed
            )
        failures.append("detached HMAC signature verification failed")

    if allow_test_trust and os.environ.get(TEST_TRUST_ENV) == "1":
        if signature_hex and trusted_key and verify_detached_hmac(
            manifest_bytes, signature_hex, trusted_key
        ):
            return CryptoVerifyResult(
                ok=True, mode="test_hmac", computed_manifest_sha256=computed
            )

    if not failures:
        failures.append(
            "no pinned digest and no valid detached signature — "
            "approval text alone is never authorization"
        )
    return CryptoVerifyResult(
        ok=False, mode="none", failures=failures, computed_manifest_sha256=computed
    )


@dataclass
class ProductionManifestAuth:
    ok: bool
    production_unlocked: bool
    failures: list[str] = field(default_factory=list)
    crypto: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "production_unlocked": self.production_unlocked,
            "failures": list(self.failures),
            "crypto": dict(self.crypto),
            "path": self.path,
        }


def load_production_manifest(
    *,
    repo_root: Path | str,
    artifact_digests: Mapping[str, str],
) -> ProductionManifestAuth:
    """Authenticate the fixed production manifest path (never artifact-controlled).

    Test HMAC success does **not** set ``production_unlocked``.
    Production unlock requires a non-None ``PINNED_EMPIRICAL_MANIFEST_SHA256``
    match or a production owner-key signature over the fixed-path manifest.
    """
    root = Path(repo_root)
    path = root / PRODUCTION_MANIFEST_RELPATH
    failures: list[str] = []
    if not path.is_file():
        return ProductionManifestAuth(
            ok=False,
            production_unlocked=False,
            failures=[
                f"production manifest missing at fixed path {PRODUCTION_MANIFEST_RELPATH} "
                "(artifact-controlled paths are refused)"
            ],
            path=str(path),
        )
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return ProductionManifestAuth(
            ok=False,
            production_unlocked=False,
            failures=[f"manifest JSON parse failed: {e}"],
            path=str(path),
        )
    if not isinstance(payload, dict):
        return ProductionManifestAuth(
            ok=False,
            production_unlocked=False,
            failures=["manifest root must be object"],
            path=str(path),
        )
    if str(payload.get("artifact_type", "")) != ARTIFACT_PROVENANCE_MANIFEST_V1:
        failures.append("manifest artifact_type mismatch")

    # Refuse circular embedding of itself into science artifacts' expected fields
    digests = payload.get("artifact_digests")
    if not isinstance(digests, Mapping):
        failures.append("manifest artifact_digests missing")
        digests = {}
    for at in REQUIRED_MANIFEST_ARTIFACT_TYPES:
        listed = str(digests.get(at, ""))
        got = str(artifact_digests.get(at, ""))
        if not got:
            failures.append(f"caller did not supply digest for {at}")
        elif listed.lower() != got.lower():
            failures.append(
                f"manifest digest for {at} does not match artifact bytes "
                f"(manifest={listed[:16]}… artifact={got[:16]}…)"
            )

    # Approval text alone is never proof
    _ = payload.get("approval_status")  # ignored as authorization
    _ = payload.get("independent_approval_record")  # explicitly not accepted

    # Signature from fixed path (not artifact-controlled)
    sig_path = root / PRODUCTION_SIGNATURE_RELPATH
    key_path = root / PRODUCTION_OWNER_KEY_RELPATH
    sig_hex = sig_path.read_text().strip() if sig_path.is_file() else None
    key_bytes = key_path.read_bytes() if key_path.is_file() else None
    canon = canonical_manifest_bytes(payload)

    crypto = verify_manifest_cryptography(
        manifest_bytes=canon,
        signature_hex=sig_hex,
        trusted_key=key_bytes,
        pinned_digest=PINNED_EMPIRICAL_MANIFEST_SHA256,
        allow_test_trust=False,
    )
    # Production unlock: only pinned digest or production key verify (not test trust)
    production_unlocked = False
    if PINNED_EMPIRICAL_MANIFEST_SHA256 and crypto.ok and crypto.mode == "pinned_digest":
        production_unlocked = True
    elif crypto.ok and crypto.mode == "detached_hmac" and key_bytes is not None:
        # Only if key came from production path and pin is set OR we have real key file
        # Without a pinned digest, a key file alone could be planted — require pin OR
        # explicit future owner process. Until pin exists, refuse unlock.
        if PINNED_EMPIRICAL_MANIFEST_SHA256:
            production_unlocked = True
        else:
            failures.append(
                "production owner key signature present but "
                "PINNED_EMPIRICAL_MANIFEST_SHA256 is unset — refuse unlock "
                "(no invented approvals)"
            )
    else:
        failures.extend(crypto.failures)
        failures.append(
            "production empirical unlock blocked: require pinned manifest digest "
            "in reviewed commit (or future owner key+pin); test trust never unlocks"
        )

    if self_generated := payload.get("self_generated"):
        if self_generated is True:
            failures.append("self_generated manifest rejected for production")
            production_unlocked = False

    ok = len(failures) == 0 and production_unlocked
    return ProductionManifestAuth(
        ok=ok,
        production_unlocked=production_unlocked,
        failures=failures,
        crypto=crypto.to_dict(),
        payload=payload,
        path=str(path),
    )


def make_test_trust_material() -> tuple[bytes, str]:
    """Isolated test-only key material (never a production owner key)."""
    key = b"TRACES_Q1_TEST_ONLY_TRUST_ANCHOR_v1_not_for_production"
    return key, "test_trust_anchor_v1"
