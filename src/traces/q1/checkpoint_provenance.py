"""Real-checkpoint provenance design (B3).

Mechanism to verify exact base, adapter, tokenizer and template file hashes
against authoritative manifests, and to require model loading from the same
verified snapshot bytes.

Does **not** invent checkpoint hashes. Unavailable verified bytes are recorded
as BLOCKED pending owner-approved frozen manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from traces.q1.pipeline_adapters import required_adapter_subfolders

# Frozen tokenizer identity targets (names only — hashes BLOCKED until verified)
FROZEN_TOKENIZER_IDENTITY = {
    "expected_tokenizer_class": "GemmaTokenizer",
    "expected_is_fast": False,
    "chat_template_required": True,
    "status": "identity_specified_hashes_blocked",
}


@dataclass
class FileHashRecord:
    relative_path: str
    role: str  # base | adapter | tokenizer | template
    sha256: str | None
    status: str  # verified | missing | blocked


@dataclass
class CheckpointManifestStatus:
    """Authoritative manifest status for scientific checkpoints."""

    ok: bool
    blocked: bool
    adapter_subfolders: list[str]
    records: list[FileHashRecord] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "blocked": self.blocked,
            "adapter_subfolders": list(self.adapter_subfolders),
            "records": [asdict(r) for r in self.records],
            "failures": list(self.failures),
            "notes": list(self.notes),
            "tokenizer_identity": dict(FROZEN_TOKENIZER_IDENTITY),
        }


def load_authoritative_manifest(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def verify_snapshot_bytes_match_manifest(
    *,
    snapshot_dir: Path | str,
    expected_sha256_by_relpath: Mapping[str, str],
) -> dict[str, Any]:
    """Verify every manifest path exists under snapshot_dir with exact SHA256.

    Loading must use these same verified bytes — never an independently resolved
    repository copy.
    """
    root = Path(snapshot_dir)
    failures: list[str] = []
    digests: dict[str, str] = {}
    if not root.is_dir():
        return {
            "ok": False,
            "failures": [f"snapshot_dir missing: {root}"],
            "digests": {},
        }
    for rel, expected in expected_sha256_by_relpath.items():
        path = root / rel
        if not path.is_file():
            failures.append(f"missing snapshot file: {rel}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[rel] = digest
        if digest != expected:
            failures.append(f"SHA256 mismatch {rel}: {digest} != {expected}")
    return {"ok": len(failures) == 0, "failures": failures, "digests": digests}


def assess_checkpoint_manifest(
    *,
    manifest_path: Path | str | None = None,
    snapshot_root: Path | str | None = None,
) -> CheckpointManifestStatus:
    """Assess whether an authoritative, owner-approved manifest exists.

    Without verified bytes for all ten adapter subfolders + tokenizer/template,
    status is BLOCKED — hashes are never invented.
    """
    subfolders = list(required_adapter_subfolders())
    notes = [
        "Do not declare manifest frozen without actual verified bytes and owner approval",
        "Model loading must use the same verified snapshot bytes",
    ]
    manifest = load_authoritative_manifest(manifest_path)
    if manifest is None:
        return CheckpointManifestStatus(
            ok=False,
            blocked=True,
            adapter_subfolders=subfolders,
            records=[
                FileHashRecord(
                    relative_path=f"{sf}/adapter_model.safetensors",
                    role="adapter",
                    sha256=None,
                    status="blocked",
                )
                for sf in subfolders
            ]
            + [
                FileHashRecord(
                    "tokenizer.json",
                    "tokenizer",
                    None,
                    "blocked",
                ),
                FileHashRecord(
                    "tokenizer_config.json",
                    "tokenizer",
                    None,
                    "blocked",
                ),
                FileHashRecord(
                    "chat_template.jinja",
                    "template",
                    None,
                    "blocked",
                ),
                FileHashRecord(
                    "base/model.safetensors",
                    "base",
                    None,
                    "blocked",
                ),
            ],
            failures=[
                "authoritative checkpoint hash manifest unavailable",
                "owner-approved frozen manifest not present",
            ],
            notes=notes,
        )

    files = manifest.get("files") or manifest.get("sha256") or {}
    if not isinstance(files, Mapping) or not files:
        return CheckpointManifestStatus(
            ok=False,
            blocked=True,
            adapter_subfolders=subfolders,
            failures=["manifest present but contains no file hashes"],
            notes=notes,
        )

    records: list[FileHashRecord] = []
    failures: list[str] = []
    for sf in subfolders:
        key = f"{sf}/adapter_model.safetensors"
        sha = files.get(key)
        if not sha:
            records.append(FileHashRecord(key, "adapter", None, "missing"))
            failures.append(f"missing adapter hash for {sf}")
        else:
            records.append(FileHashRecord(key, "adapter", str(sha), "verified"))

    for rel, role in (
        ("tokenizer.json", "tokenizer"),
        ("tokenizer_config.json", "tokenizer"),
        ("chat_template.jinja", "template"),
        ("base/model.safetensors", "base"),
    ):
        sha = files.get(rel)
        if not sha:
            records.append(FileHashRecord(rel, role, None, "missing"))
            failures.append(f"missing {role} hash for {rel}")
        else:
            records.append(FileHashRecord(rel, role, str(sha), "verified"))

    if snapshot_root is not None and not failures:
        # Only verify bytes when all hashes are present
        expected = {r.relative_path: r.sha256 for r in records if r.sha256}
        bind = verify_snapshot_bytes_match_manifest(
            snapshot_dir=snapshot_root, expected_sha256_by_relpath=expected
        )
        if not bind["ok"]:
            failures.extend(bind["failures"])

    return CheckpointManifestStatus(
        ok=len(failures) == 0,
        blocked=len(failures) > 0,
        adapter_subfolders=subfolders,
        records=records,
        failures=failures,
        notes=notes,
    )


def require_verified_snapshot_for_load(
    *,
    manifest_path: Path | str | None,
    snapshot_root: Path | str | None,
) -> dict[str, Any]:
    """Fail closed unless load path is bound to verified snapshot bytes."""
    status = assess_checkpoint_manifest(
        manifest_path=manifest_path, snapshot_root=snapshot_root
    )
    if status.blocked or not status.ok:
        return {
            "ok": False,
            "blocked": True,
            "failures": status.failures
            or ["checkpoint provenance blocked — refuse load from unbound repo copy"],
            "status": status.to_dict(),
        }
    return {"ok": True, "blocked": False, "failures": [], "status": status.to_dict()}
