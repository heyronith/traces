"""One-way empirical provenance trust (S2-1).

Empirical science artifacts must **not** embed the provenance manifest hash.
The canonical execution manifest lists artifact SHA256 digests. Production
authentication obtains the manifest from a fixed path under the trusted
executor repository root and requires an exact match to
``PINNED_EMPIRICAL_MANIFEST_SHA256``.

A repository-held symmetric HMAC key is **not** an independent owner trust
anchor and never unlocks production. Exact approval text alone is never
authorization.

Test-only pinned digests may clear an isolated validation battery but **must
never** set ``production_unlocked`` or authorize scientific execution.
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

ARTIFACT_PROVENANCE_MANIFEST_V1 = "q1_empirical_execution_provenance_manifest_v1"
TEST_TRUST_ENV = "TRACES_Q1_EMPIRICAL_TEST_TRUST"

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
    body = {
        k: v
        for k, v in payload.items()
        if k not in {"signature", "detached_signature"}
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_manifest_hmac(manifest_bytes: bytes, key: bytes) -> str:
    """Detached HMAC-SHA256 hex digest (stdlib; unit-test crypto only)."""
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
    mode: str  # "pinned_digest" | "test_hmac" | "none"
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

    If ``pinned_digest`` is configured and mismatches, fail **immediately**
    (no HMAC fallback). HMAC paths exist only for isolated unit tests when
    ``allow_test_trust=True`` and no pin is configured / pin was not supplied.
    """
    computed = sha256_bytes(manifest_bytes)

    if pinned_digest:
        if computed.lower() == str(pinned_digest).lower():
            return CryptoVerifyResult(
                ok=True, mode="pinned_digest", computed_manifest_sha256=computed
            )
        # Exact mismatch fails immediately — no HMAC bypass
        return CryptoVerifyResult(
            ok=False,
            mode="none",
            failures=[
                f"manifest digest {computed} != pinned {pinned_digest} "
                "(HMAC fallback removed; pin mismatch is fatal)"
            ],
            computed_manifest_sha256=computed,
        )

    # No pin configured: production cannot unlock via HMAC. Unit tests only.
    if allow_test_trust and os.environ.get(TEST_TRUST_ENV) == "1":
        if signature_hex and trusted_key and verify_detached_hmac(
            manifest_bytes, signature_hex, trusted_key
        ):
            return CryptoVerifyResult(
                ok=True, mode="test_hmac", computed_manifest_sha256=computed
            )
        return CryptoVerifyResult(
            ok=False,
            mode="none",
            failures=["test HMAC verification failed"],
            computed_manifest_sha256=computed,
        )

    return CryptoVerifyResult(
        ok=False,
        mode="none",
        failures=[
            "no pinned digest — approval text / repository HMAC alone is "
            "never authorization"
        ],
        computed_manifest_sha256=computed,
    )


@dataclass
class ProductionManifestAuth:
    ok: bool
    production_unlocked: bool
    battery_ok: bool = False
    failures: list[str] = field(default_factory=list)
    crypto: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    path: str | None = None
    computed_manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "production_unlocked": self.production_unlocked,
            "battery_ok": self.battery_ok,
            "failures": list(self.failures),
            "crypto": dict(self.crypto),
            "path": self.path,
            "computed_manifest_sha256": self.computed_manifest_sha256,
        }


def _load_manifest_payload(
    *, repo_root: Path | str
) -> tuple[Path, dict[str, Any] | None, bytes | None, list[str]]:
    root = Path(repo_root)
    path = root / PRODUCTION_MANIFEST_RELPATH
    if not path.is_file():
        return (
            path,
            None,
            None,
            [
                f"production manifest missing at fixed path "
                f"{PRODUCTION_MANIFEST_RELPATH} "
                "(artifact-controlled paths are refused)"
            ],
        )
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        return path, None, None, [f"manifest JSON parse failed: {e}"]
    if not isinstance(payload, dict):
        return path, None, None, ["manifest root must be object"]
    return path, payload, raw, []


def _check_artifact_digests(
    payload: Mapping[str, Any], artifact_digests: Mapping[str, str]
) -> list[str]:
    failures: list[str] = []
    if str(payload.get("artifact_type", "")) != ARTIFACT_PROVENANCE_MANIFEST_V1:
        failures.append("manifest artifact_type mismatch")
    digests = payload.get("artifact_digests")
    if not isinstance(digests, Mapping):
        failures.append("manifest artifact_digests missing")
        return failures
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
    _ = payload.get("approval_status")
    _ = payload.get("independent_approval_record")
    if payload.get("self_generated") is True:
        failures.append("self_generated manifest rejected for production")
    return failures


def load_production_manifest(
    *,
    repo_root: Path | str,
    artifact_digests: Mapping[str, str],
) -> ProductionManifestAuth:
    """Authenticate the fixed production manifest path (pin-only unlock).

    Repository HMAC keys never unlock production. ``production_unlocked``
    requires ``PINNED_EMPIRICAL_MANIFEST_SHA256`` exact match.
    """
    path, payload, _raw, load_failures = _load_manifest_payload(repo_root=repo_root)
    if payload is None:
        return ProductionManifestAuth(
            ok=False,
            production_unlocked=False,
            battery_ok=False,
            failures=load_failures,
            path=str(path),
        )

    failures = _check_artifact_digests(payload, artifact_digests)
    canon = canonical_manifest_bytes(payload)
    computed = sha256_bytes(canon)

    pin = PINNED_EMPIRICAL_MANIFEST_SHA256
    if pin is None:
        failures.append(
            "PINNED_EMPIRICAL_MANIFEST_SHA256 is unset — production empirical "
            "unlock blocked (no invented approvals; repository HMAC removed)"
        )
        return ProductionManifestAuth(
            ok=False,
            production_unlocked=False,
            battery_ok=False,
            failures=failures,
            crypto={
                "ok": False,
                "mode": "none",
                "failures": list(failures),
                "computed_manifest_sha256": computed,
            },
            payload=payload,
            path=str(path),
            computed_manifest_sha256=computed,
        )

    crypto = verify_manifest_cryptography(
        manifest_bytes=canon,
        pinned_digest=pin,
        allow_test_trust=False,
    )
    if not crypto.ok:
        failures.extend(crypto.failures)
        return ProductionManifestAuth(
            ok=False,
            production_unlocked=False,
            battery_ok=False,
            failures=failures,
            crypto=crypto.to_dict(),
            payload=payload,
            path=str(path),
            computed_manifest_sha256=computed,
        )

    production_unlocked = len(failures) == 0
    return ProductionManifestAuth(
        ok=production_unlocked,
        production_unlocked=production_unlocked,
        battery_ok=production_unlocked,
        failures=failures,
        crypto=crypto.to_dict(),
        payload=payload,
        path=str(path),
        computed_manifest_sha256=computed,
    )


def authenticate_battery_manifest(
    *,
    repo_root: Path | str,
    artifact_digests: Mapping[str, str],
    test_pinned_digest: str | None = None,
) -> ProductionManifestAuth:
    """Authenticate manifest for the empirical gate battery.

    When ``test_pinned_digest`` is provided (isolated test control), a matching
    fixed-path manifest may set ``battery_ok=True`` without ever setting
    ``production_unlocked``. Production unlock remains pin-constant only.
    """
    # Production path first
    prod = load_production_manifest(
        repo_root=repo_root, artifact_digests=artifact_digests
    )
    if prod.production_unlocked:
        return prod

    if not test_pinned_digest:
        return prod

    # Isolated test-only battery pin (never production unlock)
    path, payload, _raw, load_failures = _load_manifest_payload(repo_root=repo_root)
    if payload is None:
        return ProductionManifestAuth(
            ok=False,
            production_unlocked=False,
            battery_ok=False,
            failures=load_failures,
            path=str(path),
        )
    failures = _check_artifact_digests(payload, artifact_digests)
    # Self-generated is OK for isolated test battery fixtures only
    failures = [f for f in failures if "self_generated" not in f]
    canon = canonical_manifest_bytes(payload)
    crypto = verify_manifest_cryptography(
        manifest_bytes=canon,
        pinned_digest=test_pinned_digest,
        allow_test_trust=False,
    )
    if not crypto.ok:
        failures.extend(crypto.failures)
    battery_ok = len(failures) == 0 and crypto.ok
    return ProductionManifestAuth(
        ok=False,  # never production-ok via test pin
        production_unlocked=False,
        battery_ok=battery_ok,
        failures=failures
        if not battery_ok
        else [
            "test_pinned_digest cleared battery only; "
            "production_unlocked remains False"
        ],
        crypto=crypto.to_dict(),
        payload=payload,
        path=str(path),
        computed_manifest_sha256=crypto.computed_manifest_sha256,
    )


def make_test_trust_material() -> tuple[bytes, str]:
    """Isolated test-only key material (never a production owner key)."""
    key = b"TRACES_Q1_TEST_ONLY_TRUST_ANCHOR_v1_not_for_production"
    return key, "test_trust_anchor_v1"
