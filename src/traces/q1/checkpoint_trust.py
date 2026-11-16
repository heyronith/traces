"""Owner-anchored checkpoint provenance trust (D5-A).

Production identity requires a separately owner-reviewed manifest whose
canonical digest matches ``PINNED_CHECKPOINT_MANIFEST_SHA256``.

A Git commit path alone is never authorization. Cursor cannot self-approve
by writing a manifest under the repo. The scientific pin remains **unset**
until an independent owner freeze supplies a real digest — do not invent one.

Engineering/synthetic manifests may authenticate only under the isolated
engineering interface and **never** set ``production_unlocked``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from traces.q1.pipeline_adapters import required_adapter_subfolders

# Production pin — empty until independently reviewed owner freeze.
# Never invent a real digest, approval, or checkpoint hash.
PINNED_CHECKPOINT_MANIFEST_SHA256: str | None = None

PRODUCTION_CHECKPOINT_MANIFEST_RELPATH = (
    "results/q1_implementation_prep/CHECKPOINT_PROVENANCE_MANIFEST.json"
)

CHECKPOINT_MANIFEST_V1 = "q1_checkpoint_provenance_manifest_v1"
ENGINEERING_CHECKPOINT_TRUST_ENV = "TRACES_Q1_CHECKPOINT_ENGINEERING_TRUST"

# Adapter weight file basename used for pairwise-distinct ten-identity check.
ADAPTER_WEIGHT_BASENAME = "adapter_model.safetensors"

# Exact load-relevant inventory (D5-B). No silent substitutions.
REQUIRED_ADAPTER_LOAD_FILES: tuple[str, ...] = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)

REQUIRED_BASE_LOAD_FILES: tuple[str, ...] = (
    "model.safetensors",
    "config.json",
)

# Competing / untrusted tokenizer sources — must not be present for load.
FORBIDDEN_ADAPTER_LOAD_FILES: frozenset[str] = frozenset(
    {
        "tokenizer.json",
        "tokenizer.model.json",
    }
)

# Filename suffixes treated as loadable (must be listed in the unit inventory).
LOADABLE_SUFFIXES: frozenset[str] = frozenset(
    {".safetensors", ".bin", ".pt", ".json", ".model", ".jinja"}
)

TRUSTED_MANIFEST_TOP_KEYS: frozenset[str] = frozenset(
    {
        "artifact_type",
        "self_generated",
        "units",
        "base_files",
        "notes",
        "signature",
        "detached_signature",
    }
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_checkpoint_manifest_bytes(payload: Mapping[str, Any]) -> bytes:
    body = {
        k: v
        for k, v in payload.items()
        if k not in {"signature", "detached_signature"}
    }
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _reject_manifest_relpath(rel: str) -> None:
    if not rel or not str(rel).strip():
        raise ValueError("empty manifest path")
    p = Path(rel)
    if p.is_absolute():
        raise ValueError(f"absolute path forbidden in manifest: {rel!r}")
    if ".." in p.parts:
        raise ValueError(f"path traversal forbidden in manifest: {rel!r}")
    if any(part.startswith("\\") for part in p.parts):
        raise ValueError(f"malformed manifest path: {rel!r}")


def validate_checkpoint_manifest_schema(
    payload: Mapping[str, Any],
) -> list[str]:
    """Structural + trust-key validation (no invented hashes)."""
    failures: list[str] = []
    if not isinstance(payload, Mapping):
        return ["checkpoint manifest root must be object"]
    for key in payload:
        if key not in TRUSTED_MANIFEST_TOP_KEYS:
            failures.append(f"unexpected / untrusted manifest key: {key!r}")
    if payload.get("artifact_type") != CHECKPOINT_MANIFEST_V1:
        failures.append(
            f"artifact_type must be {CHECKPOINT_MANIFEST_V1!r} "
            f"(got {payload.get('artifact_type')!r})"
        )
    if payload.get("self_generated") is True:
        failures.append("self_generated checkpoint manifest rejected")

    base_files = payload.get("base_files")
    if not isinstance(base_files, Mapping) or not base_files:
        failures.append("base_files missing or empty")
    else:
        for key, digest in base_files.items():
            try:
                if not str(key).startswith("base:"):
                    failures.append(f"base_files key must be role-tagged base:: {key!r}")
                    continue
                rel = str(key).split(":", 1)[1]
                _reject_manifest_relpath(rel)
            except ValueError as e:
                failures.append(str(e))
            if not isinstance(digest, str) or len(digest) != 64:
                failures.append(f"malformed base digest for {key!r}")

    units = payload.get("units")
    if not isinstance(units, Mapping):
        failures.append("units mapping missing")
        return failures

    required = set(required_adapter_subfolders())
    got = set(units.keys())
    if got != required:
        failures.append(
            f"units must cover exactly the ten adapter identities; "
            f"missing={sorted(required - got)} extra={sorted(got - required)}"
        )

    adapter_weight_digests: dict[str, str] = {}
    for sf, unit in units.items():
        if not isinstance(unit, Mapping):
            failures.append(f"unit {sf!r} must be object")
            continue
        family = unit.get("family")
        seed = unit.get("seed")
        subfolder = unit.get("subfolder")
        files = unit.get("files")
        if subfolder != sf:
            failures.append(
                f"unit {sf!r} subfolder field {subfolder!r} mismatches key"
            )
        expected_family, seed_tok = sf.split("/", 1)
        if family != expected_family:
            failures.append(f"unit {sf!r} family {family!r} mismatches")
        try:
            expected_seed = int(seed_tok.removeprefix("seed"))
        except ValueError:
            failures.append(f"unit {sf!r} unparseable seed token")
            expected_seed = None
        if seed != expected_seed:
            failures.append(f"unit {sf!r} seed {seed!r} mismatches key")
        if not isinstance(files, Mapping) or not files:
            failures.append(f"unit {sf!r} files missing")
            continue
        for key, digest in files.items():
            try:
                if not str(key).startswith("adapter:"):
                    failures.append(
                        f"unit {sf!r} file key must be adapter:-tagged: {key!r}"
                    )
                    continue
                rel = str(key).split(":", 1)[1]
                _reject_manifest_relpath(rel)
                if not rel.startswith(sf.rstrip("/") + "/") and rel != sf:
                    failures.append(
                        f"unit {sf!r} file escapes subfolder: {key!r}"
                    )
            except ValueError as e:
                failures.append(str(e))
            if not isinstance(digest, str) or len(digest) != 64:
                failures.append(f"malformed adapter digest for {key!r}")
        weight_key = f"adapter:{sf}/{ADAPTER_WEIGHT_BASENAME}"
        if weight_key not in files:
            failures.append(f"unit {sf!r} missing {weight_key}")
        else:
            adapter_weight_digests[sf] = str(files[weight_key])
        # Required load files must be listed
        for name in REQUIRED_ADAPTER_LOAD_FILES:
            need = f"adapter:{sf}/{name}"
            if need not in files:
                failures.append(f"unit {sf!r} missing required load file {need}")

    # Pairwise distinct adapter-weight identities (shared base/tokenizer OK)
    seen: dict[str, str] = {}
    for sf, digest in adapter_weight_digests.items():
        if digest in seen:
            failures.append(
                f"adapter-weight digests not pairwise distinct: "
                f"{sf} collides with {seen[digest]} (D5-A)"
            )
        else:
            seen[digest] = sf
    if len(adapter_weight_digests) == 10 and len(seen) != 10:
        failures.append(
            "ten adapter-weight identities must be pairwise distinct (D5-A)"
        )

    return failures


@dataclass
class CheckpointManifestAuth:
    ok: bool
    production_unlocked: bool
    engineering_only: bool
    failures: list[str] = field(default_factory=list)
    payload: dict[str, Any] | None = None
    path: str | None = None
    computed_manifest_sha256: str | None = None
    mode: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "production_unlocked": self.production_unlocked,
            "engineering_only": self.engineering_only,
            "failures": list(self.failures),
            "path": self.path,
            "computed_manifest_sha256": self.computed_manifest_sha256,
            "mode": self.mode,
            "pin_configured": PINNED_CHECKPOINT_MANIFEST_SHA256 is not None,
        }


def unit_digests_from_auth(
    auth: CheckpointManifestAuth, *, subfolder: str
) -> dict[str, str]:
    """Merge base_files + unit files for one subfolder (role-tagged keys)."""
    if auth.payload is None:
        raise RuntimeError("checkpoint manifest payload unavailable")
    units = auth.payload.get("units") or {}
    if subfolder not in units:
        raise RuntimeError(
            f"authoritative manifest missing unit {subfolder!r} (D5-A)"
        )
    out: dict[str, str] = {}
    base_files = auth.payload.get("base_files") or {}
    for k, v in base_files.items():
        out[str(k)] = str(v)
    for k, v in (units[subfolder].get("files") or {}).items():
        out[str(k)] = str(v)
    return out


def authenticate_production_checkpoint_manifest(
    *,
    repo_root: Path | str,
) -> CheckpointManifestAuth:
    """Fail-closed production unlock via owner pin only (D5-A).

    Caller-supplied dicts are never accepted here. Pin unset → blocked.
    """
    root = Path(repo_root)
    path = root / PRODUCTION_CHECKPOINT_MANIFEST_RELPATH
    pin = PINNED_CHECKPOINT_MANIFEST_SHA256
    if pin is None:
        return CheckpointManifestAuth(
            ok=False,
            production_unlocked=False,
            engineering_only=False,
            failures=[
                "PINNED_CHECKPOINT_MANIFEST_SHA256 is unset — production "
                "checkpoint identity blocked (D5-A; no invented owner pin; "
                "caller expected_weight_file_sha256 is not authoritative)"
            ],
            path=str(path),
            mode="pin_unset",
        )
    if not path.is_file():
        return CheckpointManifestAuth(
            ok=False,
            production_unlocked=False,
            engineering_only=False,
            failures=[
                f"authoritative checkpoint manifest missing at fixed path {path}"
            ],
            path=str(path),
            mode="missing_file",
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return CheckpointManifestAuth(
            ok=False,
            production_unlocked=False,
            engineering_only=False,
            failures=[f"checkpoint manifest JSON parse failed: {e}"],
            path=str(path),
            mode="parse_error",
        )
    if not isinstance(payload, dict):
        return CheckpointManifestAuth(
            ok=False,
            production_unlocked=False,
            engineering_only=False,
            failures=["checkpoint manifest root must be object"],
            path=str(path),
            mode="schema",
        )
    failures = validate_checkpoint_manifest_schema(payload)
    canon = canonical_checkpoint_manifest_bytes(payload)
    computed = sha256_bytes(canon)
    if computed.lower() != str(pin).lower():
        failures.append(
            f"checkpoint manifest digest {computed} != owner pin {pin} (D5-A)"
        )
    unlocked = len(failures) == 0
    return CheckpointManifestAuth(
        ok=unlocked,
        production_unlocked=unlocked,
        engineering_only=False,
        failures=failures,
        payload=payload if unlocked else None,
        path=str(path),
        computed_manifest_sha256=computed,
        mode="pinned_digest" if unlocked else "pin_mismatch",
    )


def authenticate_engineering_checkpoint_manifest(
    payload: Mapping[str, Any],
    *,
    expected_digest: str | None = None,
) -> CheckpointManifestAuth:
    """Isolated engineering-test authentication (never production_unlocked).

    Requires ``TRACES_Q1_CHECKPOINT_ENGINEERING_TRUST=1`` so synthetic fixtures
    cannot silently authorize production progression.
    """
    if os.environ.get(ENGINEERING_CHECKPOINT_TRUST_ENV) != "1":
        return CheckpointManifestAuth(
            ok=False,
            production_unlocked=False,
            engineering_only=True,
            failures=[
                f"{ENGINEERING_CHECKPOINT_TRUST_ENV}=1 required for engineering "
                "checkpoint manifests (isolated from production)"
            ],
            mode="engineering_env_missing",
        )
    if not isinstance(payload, Mapping):
        return CheckpointManifestAuth(
            ok=False,
            production_unlocked=False,
            engineering_only=True,
            failures=["engineering manifest root must be object"],
            mode="schema",
        )
    # Engineering payloads may set self_generated for clarity; strip for schema
    body = dict(payload)
    body.pop("self_generated", None)
    body.setdefault("artifact_type", CHECKPOINT_MANIFEST_V1)
    failures = validate_checkpoint_manifest_schema(body)
    canon = canonical_checkpoint_manifest_bytes(body)
    computed = sha256_bytes(canon)
    if expected_digest is not None and computed.lower() != str(expected_digest).lower():
        failures.append(
            f"engineering manifest digest {computed} != expected {expected_digest}"
        )
    ok = len(failures) == 0
    return CheckpointManifestAuth(
        ok=ok,
        production_unlocked=False,  # NEVER
        engineering_only=True,
        failures=failures,
        payload=dict(body) if ok else None,
        computed_manifest_sha256=computed,
        mode="engineering_synthetic",
    )


def build_engineering_ten_unit_manifest(
    *,
    base_files: Mapping[str, str],
    unit_files_by_subfolder: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Helper to assemble a schema-shaped engineering manifest (tests only)."""
    units: dict[str, Any] = {}
    for sf, files in unit_files_by_subfolder.items():
        family, seed_tok = sf.split("/", 1)
        units[sf] = {
            "family": family,
            "seed": int(seed_tok.removeprefix("seed")),
            "subfolder": sf,
            "files": dict(files),
        }
    return {
        "artifact_type": CHECKPOINT_MANIFEST_V1,
        "self_generated": False,
        "base_files": dict(base_files),
        "units": units,
        "notes": ["engineering synthetic — not production authorization"],
    }
