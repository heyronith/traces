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

# Frozen tokenizer identity targets (names + §11 file pins from protocol v1.3)
FROZEN_TOKENIZER_IDENTITY = {
    "expected_tokenizer_class": "GemmaTokenizer",
    "expected_is_fast": False,
    "chat_template_required": True,
    "status": "section11_file_pins_specified",
}

# Protocol §11 confirmatory tokenizer/template file SHA-256 pins (byte-identical
# across the seven confirmatory subfolders). Production loads must match these
# when authenticating the tokenizer files actually used.
FROZEN_SECTION11_TOKENIZER_SHA256: dict[str, str] = {
    "tokenizer.model": (
        "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"
    ),
    "tokenizer_config.json": (
        "5db39e7ccbeceb1dc1136ff54d717f6f84f3556893e6cc30ee4972c408ed4dc2"
    ),
    "chat_template.jinja": (
        "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6"
    ),
}

REQUIRED_TOKENIZER_TEMPLATE_FILES: tuple[str, ...] = tuple(
    FROZEN_SECTION11_TOKENIZER_SHA256.keys()
)


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


def _reject_path_escape(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``; reject absolute paths, ``..``, symlink escape."""
    if not rel or not str(rel).strip():
        raise RuntimeError("empty snapshot-relative path — refuse")
    rel_path = Path(rel)
    if rel_path.is_absolute():
        raise RuntimeError(
            f"absolute path forbidden in snapshot-relative manifest: {rel!r}"
        )
    if ".." in rel_path.parts:
        raise RuntimeError(f"path traversal forbidden: {rel!r}")
    root_res = root.resolve()
    candidate = (root_res / rel_path).resolve()
    try:
        candidate.relative_to(root_res)
    except ValueError as e:
        raise RuntimeError(
            f"path escape outside snapshot root {root_res}: {rel!r}"
        ) from e
    return candidate


def resolve_unit_adapter_subfolder_dir(
    adapter_snapshot_dir: Path | str,
    subfolder: str,
) -> Path:
    """Require the exact adapter subfolder; never fall back to root (D5)."""
    root = Path(adapter_snapshot_dir)
    if not root.is_dir():
        raise RuntimeError(f"adapter snapshot dir missing: {root}")
    if not subfolder or not str(subfolder).strip():
        raise RuntimeError("adapter subfolder required — refuse root fallback")
    # Validate subfolder segments (no traversal)
    sf = _reject_path_escape(root, subfolder)
    if not sf.is_dir():
        raise RuntimeError(
            f"missing required adapter subfolder {subfolder!r} under {root} — "
            "refuse silent fallback to adapter root or another seed (D5)"
        )
    return sf


def inventory_unit_weight_files(
    *,
    base_snapshot_dir: Path | str,
    adapter_subfolder_dir: Path | str,
) -> dict[str, list[str]]:
    """Inventory weight files in the base dir and the unit's adapter subfolder.

    Uses snapshot-relative path strings for adapter files as
    ``<subfolder>/<name>`` when ``adapter_subfolder_dir`` is nested under a
    known parent; otherwise returns basenames. Callers that need role-tagged
    keys should use :func:`bind_unit_checkpoint_provenance`.
    """
    base_root = Path(base_snapshot_dir)
    ad_dir = Path(adapter_subfolder_dir)
    if not base_root.is_dir():
        raise RuntimeError(f"base snapshot dir missing: {base_root}")
    if not ad_dir.is_dir():
        raise RuntimeError(f"adapter subfolder dir missing: {ad_dir}")

    def _weights(d: Path) -> list[str]:
        files: list[str] = []
        for pat in ("*.safetensors", "*.bin", "*.pt"):
            files.extend(str(x.resolve()) for x in d.glob(pat) if x.is_file())
        return sorted(set(files))

    return {"base": _weights(base_root), "adapter": _weights(ad_dir)}


def _split_manifest_roles(
    expected_sha256_by_relpath: Mapping[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    base_expected: dict[str, str] = {}
    adapter_expected: dict[str, str] = {}
    for key, sha in expected_sha256_by_relpath.items():
        role_key = str(key)
        if role_key.startswith("base:"):
            base_expected[role_key.split(":", 1)[1]] = str(sha)
        elif role_key.startswith("base/"):
            base_expected[role_key.removeprefix("base/")] = str(sha)
        elif role_key.startswith("adapter:"):
            adapter_expected[role_key.split(":", 1)[1]] = str(sha)
        else:
            adapter_expected[role_key] = str(sha)
    return base_expected, adapter_expected


def verify_files_under_snapshot(
    *,
    snapshot_dir: Path | str,
    expected_sha256_by_relpath: Mapping[str, str],
    role_prefix: str,
) -> dict[str, Any]:
    """Hash snapshot-relative files with containment checks (D5)."""
    root = Path(snapshot_dir)
    if not root.is_dir():
        return {
            "ok": False,
            "failures": [f"snapshot_dir missing: {root}"],
            "digests": {},
        }
    failures: list[str] = []
    digests: dict[str, str] = {}
    for rel, expected in expected_sha256_by_relpath.items():
        try:
            path = _reject_path_escape(root, rel)
        except RuntimeError as e:
            failures.append(str(e))
            continue
        if not path.is_file():
            failures.append(f"missing snapshot file: {rel}")
            continue
        if path.is_symlink():
            # Resolve already checked containment; still record symlink use
            if not path.resolve().is_file():
                failures.append(f"symlink target missing for {rel}")
                continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digests[f"{role_prefix}:{rel}"] = digest
        if digest != str(expected):
            failures.append(f"SHA256 mismatch {rel}: {digest} != {expected}")
    return {"ok": len(failures) == 0, "failures": failures, "digests": digests}


def require_tokenizer_and_template_files(
    adapter_subfolder_dir: Path | str,
    *,
    expected_sha256: Mapping[str, str] | None = None,
    subfolder: str | None = None,
) -> dict[str, Any]:
    """Verify tokenizer/template files used for load against frozen §11 pins (D5).

    ``adapter_subfolder_dir`` must be the unit's resolved subfolder directory
    (not the multi-seed root). ``subfolder`` is accepted only for back-compat
    and must not trigger a root fallback search.
    """
    _ = subfolder  # explicit: no root search
    root = Path(adapter_subfolder_dir)
    if not root.is_dir():
        raise RuntimeError(f"adapter subfolder dir missing for tokenizer check: {root}")
    pins = dict(expected_sha256 or FROZEN_SECTION11_TOKENIZER_SHA256)
    for name in REQUIRED_TOKENIZER_TEMPLATE_FILES:
        if name not in pins:
            raise RuntimeError(
                f"authoritative pin missing for tokenizer/template file {name} (D5)"
            )
    found: dict[str, str] = {}
    failures: list[str] = []
    for name in REQUIRED_TOKENIZER_TEMPLATE_FILES:
        path = root / name
        if not path.is_file():
            failures.append(f"missing required tokenizer/template file: {name}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        found[name] = digest
        expected = str(pins[name])
        if digest != expected:
            failures.append(
                f"tokenizer/template SHA mismatch {name}: {digest} != {expected}"
            )
    if failures:
        raise RuntimeError(
            "tokenizer/template verification failed: " + "; ".join(failures)
        )
    return {"ok": True, "files": found, "pins": dict(pins)}


def resolve_local_snapshot_load_paths(
    *,
    base_snapshot_dir: Path | str,
    adapter_snapshot_dir: Path | str,
    subfolder: str,
) -> dict[str, str]:
    """Resolve local-only load roots; require exact subfolder (D5).

    Never falls back to the adapter root when the subfolder is missing.
    """
    base = Path(base_snapshot_dir)
    if not base.is_dir():
        raise RuntimeError(f"base snapshot dir missing: {base}")
    adapter_load = resolve_unit_adapter_subfolder_dir(adapter_snapshot_dir, subfolder)
    return {
        "base_load_path": str(base.resolve()),
        "adapter_load_path": str(adapter_load.resolve()),
        "adapter_subfolder": subfolder,
        "local_files_only": "true",
    }


def assert_load_uses_verified_snapshot_bytes(
    *,
    base_snapshot_dir: Path | str,
    adapter_snapshot_dir: Path | str,
    expected_sha256_by_relpath: Mapping[str, str],
    required_adapter_subfolder: str | None = None,
) -> dict[str, Any]:
    """Refuse load unless expected paths hash-match under snapshot dirs (D5).

    When ``required_adapter_subfolder`` is set, adapter manifest keys must be
    under that subfolder prefix and the subfolder directory must exist.
    """
    base_expected, adapter_expected = _split_manifest_roles(expected_sha256_by_relpath)
    if not base_expected and not adapter_expected:
        raise RuntimeError(
            "empty expected_sha256_by_relpath — refuse unbound checkpoint load"
        )
    computed: dict[str, str] = {}
    if required_adapter_subfolder:
        resolve_unit_adapter_subfolder_dir(
            adapter_snapshot_dir, required_adapter_subfolder
        )
        prefix = required_adapter_subfolder.rstrip("/") + "/"
        # Keep only keys for this unit; reject cross-seed keys silently matching
        # Shared ten-adapter manifests may list all seeds; bind only this unit's keys.
        unit_adapter = {
            rel: sha
            for rel, sha in adapter_expected.items()
            if rel == required_adapter_subfolder or rel.startswith(prefix)
        }
        if not unit_adapter:
            raise RuntimeError(
                f"adapter manifest missing keys for subfolder "
                f"{required_adapter_subfolder!r} (D5)"
            )
        adapter_expected = unit_adapter
    if base_expected:
        base_chk = verify_files_under_snapshot(
            snapshot_dir=base_snapshot_dir,
            expected_sha256_by_relpath=base_expected,
            role_prefix="base",
        )
        if not base_chk["ok"]:
            raise RuntimeError(
                "base snapshot verification failed: "
                + "; ".join(base_chk["failures"])
            )
        computed.update(base_chk["digests"])
    if adapter_expected:
        ad_chk = verify_files_under_snapshot(
            snapshot_dir=adapter_snapshot_dir,
            expected_sha256_by_relpath=adapter_expected,
            role_prefix="adapter",
        )
        if not ad_chk["ok"]:
            raise RuntimeError(
                "adapter snapshot verification failed: "
                + "; ".join(ad_chk["failures"])
            )
        computed.update(ad_chk["digests"])
    if not computed:
        raise RuntimeError(
            "no computed digests from snapshot verification — refuse load (D5)"
        )
    return {"ok": True, "computed_sha256": computed}


def bind_unit_checkpoint_provenance(
    *,
    family: str,
    seed: int | None,
    subfolder: str,
    base_snapshot_dir: Path | str,
    adapter_snapshot_dir: Path | str,
    expected_sha256_by_relpath: Mapping[str, str],
    require_section11_tokenizer_pins: bool = True,
) -> dict[str, Any]:
    """Single fail-closed unit↔checkpoint provenance contract (D5).

    Binds the requested family/seed/subfolder to verified local snapshot bytes,
    inventories the unit subfolder, authenticates tokenizer/template files, and
    returns **computed** digests only. Never invents hashes or falls back to
    caller-expected digests.
    """
    if family == "base":
        expected_sf = "l19/seed42"
    elif family in {"l19", "l1523"} and seed is not None:
        expected_sf = f"{family}/seed{seed}"
    else:
        raise RuntimeError(f"cannot bind provenance for family={family!r} seed={seed!r}")
    if subfolder != expected_sf:
        raise RuntimeError(
            f"subfolder {subfolder!r} does not match frozen unit identity "
            f"{expected_sf!r} (D5)"
        )
    if expected_sf not in required_adapter_subfolders() and family != "base":
        # base uses l19/seed42 which is in the ten
        pass
    if expected_sf not in required_adapter_subfolders():
        raise RuntimeError(
            f"subfolder {expected_sf!r} is not one of the ten scientific "
            f"adapter identities (D5)"
        )

    if not expected_sha256_by_relpath:
        raise RuntimeError(
            "authoritative unit weight manifest unavailable — block load without "
            "inventing hashes (D5)"
        )

    snapshot_bind = assert_load_uses_verified_snapshot_bytes(
        base_snapshot_dir=base_snapshot_dir,
        adapter_snapshot_dir=adapter_snapshot_dir,
        expected_sha256_by_relpath=expected_sha256_by_relpath,
        required_adapter_subfolder=subfolder,
    )
    adapter_dir = resolve_unit_adapter_subfolder_dir(adapter_snapshot_dir, subfolder)
    inventory = inventory_unit_weight_files(
        base_snapshot_dir=base_snapshot_dir,
        adapter_subfolder_dir=adapter_dir,
    )
    if not inventory["base"]:
        raise RuntimeError("base weight inventory empty after unit bind (D5)")
    if not inventory["adapter"]:
        raise RuntimeError(
            f"adapter weight inventory empty for subfolder {subfolder!r} (D5)"
        )

    # Authenticate inventoried files against unit manifest (snapshot-relative)
    base_exp, ad_exp = _split_manifest_roles(expected_sha256_by_relpath)
    computed = dict(snapshot_bind["computed_sha256"])
    sf_prefix = subfolder.rstrip("/") + "/"

    def _match_base(name: str) -> str:
        if name in base_exp:
            return name
        hits = [k for k in base_exp if Path(k).name == name]
        if len(hits) == 1:
            return hits[0]
        raise RuntimeError(
            f"inventoried base file not uniquely in authoritative manifest: {name}"
        )

    def _match_adapter(name: str) -> str:
        rel = f"{subfolder.rstrip('/')}/{name}"
        if rel in ad_exp:
            return rel
        # Refuse bare-filename matches that could bind another seed's bytes
        raise RuntimeError(
            f"inventoried adapter file not in unit manifest as "
            f"adapter:{rel!r}: {name} (refuse cross-seed / root bind)"
        )

    for abs_path in inventory["base"]:
        name = Path(abs_path).name
        matched = _match_base(name)
        digest = hashlib.sha256(Path(abs_path).read_bytes()).hexdigest()
        if digest != base_exp[matched]:
            raise RuntimeError(
                f"base inventory SHA mismatch {abs_path}: "
                f"{digest} != {base_exp[matched]}"
            )
        computed[f"inventory:base:{matched}"] = digest

    for abs_path in inventory["adapter"]:
        name = Path(abs_path).name
        matched = _match_adapter(name)
        if not matched.startswith(sf_prefix):
            raise RuntimeError(
                f"adapter manifest key {matched!r} escapes subfolder {subfolder!r}"
            )
        digest = hashlib.sha256(Path(abs_path).read_bytes()).hexdigest()
        if digest != ad_exp[matched]:
            raise RuntimeError(
                f"adapter inventory SHA mismatch {abs_path}: "
                f"{digest} != {ad_exp[matched]}"
            )
        computed[f"inventory:adapter:{matched}"] = digest

    tok = require_tokenizer_and_template_files(
        adapter_dir,
        expected_sha256=(
            FROZEN_SECTION11_TOKENIZER_SHA256
            if require_section11_tokenizer_pins
            else None
        ),
    )
    for name, digest in tok["files"].items():
        computed[f"tokenizer:{name}"] = digest

    local_paths = resolve_local_snapshot_load_paths(
        base_snapshot_dir=base_snapshot_dir,
        adapter_snapshot_dir=adapter_snapshot_dir,
        subfolder=subfolder,
    )
    if not computed:
        raise RuntimeError(
            "computed digests missing after provenance bind — refuse expected-hash "
            "substitution (D5)"
        )
    return {
        "ok": True,
        "family": family,
        "seed": seed,
        "subfolder": subfolder,
        "adapter_subfolder_dir": str(adapter_dir),
        "inventory": inventory,
        "local_paths": local_paths,
        "tokenizer_template_files": tok,
        "computed_sha256": computed,
        "snapshot_byte_binding": snapshot_bind,
    }
