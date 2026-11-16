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

from traces.q1.checkpoint_trust import required_adapter_subfolders

# Frozen tokenizer identity targets (names + §11 file pins from protocol v1.3)
FROZEN_TOKENIZER_IDENTITY = {
    "expected_tokenizer_class": "GemmaTokenizer",
    "expected_is_fast": False,
    "chat_template_required": True,
    "status": "section11_file_pins_specified",
}

# Protocol §11 confirmatory tokenizer/template file SHA-256 pins (byte-identical
# across the seven confirmatory subfolders). Production loads must match these
# when authenticating the tokenizer files actually used. Do not alter these
# frozen digests. Includes all four §11 files (D5-B).
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
    "special_tokens_map.json": (
        "5b36db1ae0240191639152e5edfc68785230efb3b3cbb3ba546687a5a9e9c680"
    ),
}

REQUIRED_TOKENIZER_TEMPLATE_FILES: tuple[str, ...] = tuple(
    FROZEN_SECTION11_TOKENIZER_SHA256.keys()
)
assert set(REQUIRED_TOKENIZER_TEMPLATE_FILES) == {
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
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


def _reject_symlink(path: Path, *, label: str) -> None:
    """Refuse directory or file symlinks, including cross-seed links (D5-A)."""
    if path.is_symlink():
        raise RuntimeError(
            f"symlink refused for {label}: {path} — refuse cross-seed / "
            "escape via symlink (D5-A)"
        )
    # Also refuse if any path component under the leaf is a symlink
    try:
        for parent in [path, *path.parents]:
            if parent.is_symlink():
                raise RuntimeError(
                    f"symlink in path refused for {label}: {parent} (D5-A)"
                )
    except RuntimeError:
        raise
    except OSError as e:
        raise RuntimeError(f"symlink inspection failed for {label}: {e}") from e


def resolve_unit_adapter_subfolder_dir(
    adapter_snapshot_dir: Path | str,
    subfolder: str,
) -> Path:
    """Require the exact adapter subfolder; never fall back to root (D5)."""
    root = Path(adapter_snapshot_dir)
    if not root.is_dir():
        raise RuntimeError(f"adapter snapshot dir missing: {root}")
    _reject_symlink(root, label="adapter snapshot root")
    if not subfolder or not str(subfolder).strip():
        raise RuntimeError("adapter subfolder required — refuse root fallback")
    # Validate subfolder segments (no traversal) before existence check
    rel_path = Path(subfolder)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise RuntimeError(f"path traversal forbidden in subfolder: {subfolder!r}")
    # Walk components and refuse symlinks before resolve() follows them
    cur = root
    for part in rel_path.parts:
        cur = cur / part
        if cur.is_symlink():
            raise RuntimeError(
                f"symlink refused for adapter subfolder component {part!r} "
                f"under {subfolder!r} (D5-A)"
            )
        if not cur.exists():
            raise RuntimeError(
                f"missing required adapter subfolder {subfolder!r} under {root} — "
                "refuse silent fallback to adapter root or another seed (D5)"
            )
    if not cur.is_dir():
        raise RuntimeError(
            f"missing required adapter subfolder {subfolder!r} under {root} — "
            "refuse silent fallback to adapter root or another seed (D5)"
        )
    _reject_symlink(cur, label=f"adapter subfolder {subfolder!r}")
    return cur


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
    _reject_symlink(base_root, label="base snapshot")
    _reject_symlink(ad_dir, label="adapter subfolder")

    def _weights(d: Path) -> list[str]:
        files: list[str] = []
        for pat in ("*.safetensors", "*.bin", "*.pt"):
            for x in d.glob(pat):
                if not x.is_file():
                    continue
                if x.is_symlink():
                    raise RuntimeError(
                        f"symlink refused in weight inventory: {x} (D5-A)"
                    )
                files.append(str(x.resolve()))
        return sorted(set(files))

    return {"base": _weights(base_root), "adapter": _weights(ad_dir)}


def inventory_load_relevant_files(
    *,
    base_snapshot_dir: Path | str,
    adapter_subfolder_dir: Path | str,
) -> dict[str, list[str]]:
    """Inventory every load-relevant file (weights + configs + tokenizer) (D5-B)."""
    from traces.q1.checkpoint_trust import LOADABLE_SUFFIXES

    base_root = Path(base_snapshot_dir)
    ad_dir = Path(adapter_subfolder_dir)
    _reject_symlink(base_root, label="base snapshot")
    _reject_symlink(ad_dir, label="adapter subfolder")

    def _loadable(d: Path) -> list[str]:
        out: list[str] = []
        for x in sorted(d.iterdir()):
            if not x.is_file():
                continue
            if x.is_symlink():
                raise RuntimeError(
                    f"symlink refused in load-relevant inventory: {x} (D5-A)"
                )
            if x.suffix in LOADABLE_SUFFIXES or x.name.endswith(".model"):
                out.append(str(x.resolve()))
        return out

    return {"base": _loadable(base_root), "adapter": _loadable(ad_dir)}


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
            # Containment without following symlinks for the leaf check
            rel_path = Path(rel)
            if rel_path.is_absolute() or ".." in rel_path.parts:
                raise RuntimeError(f"path traversal forbidden: {rel!r}")
            path = root / rel_path
            if path.is_symlink():
                raise RuntimeError(
                    f"symlink refused for snapshot file {rel!r} (D5-A)"
                )
            path = _reject_path_escape(root, rel)
            _reject_symlink(path, label=f"{role_prefix}:{rel}")
        except RuntimeError as e:
            failures.append(str(e))
            continue
        if not path.is_file():
            failures.append(f"missing snapshot file: {rel}")
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
    _reject_symlink(root, label="tokenizer adapter subfolder")
    from traces.q1.checkpoint_trust import FORBIDDEN_ADAPTER_LOAD_FILES

    for banned in FORBIDDEN_ADAPTER_LOAD_FILES:
        banned_path = root / banned
        if banned_path.exists():
            raise RuntimeError(
                f"competing tokenizer source present: {banned} — refuse "
                "unverified tokenizer.json / alternate tokenizer (D5-B)"
            )
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
        if path.is_symlink():
            failures.append(f"symlink refused for tokenizer/template file: {name}")
            continue
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


def _verify_adapter_config_scaling(adapter_dir: Path, *, computed: dict[str, str]) -> dict[str, Any]:
    """Authenticate adapter_config.json bytes and record LoRA scaling fields (D5-B)."""
    cfg_path = adapter_dir / "adapter_config.json"
    if cfg_path.is_symlink():
        raise RuntimeError("symlink refused for adapter_config.json (D5-A)")
    if not cfg_path.is_file():
        raise RuntimeError("adapter_config.json missing — refuse load (D5-B)")
    raw = cfg_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    computed["adapter_config.json"] = digest
    try:
        cfg = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise RuntimeError(f"adapter_config.json unreadable: {e}") from e
    if not isinstance(cfg, dict):
        raise RuntimeError("adapter_config.json must be a JSON object")
    if cfg.get("use_topk") is True:
        raise RuntimeError("Q1 refuses use_topk=true adapters")
    if "r" not in cfg or "lora_alpha" not in cfg:
        raise RuntimeError(
            "adapter_config.json missing LoRA scaling fields r / lora_alpha (D5-B)"
        )
    try:
        r = int(cfg["r"])
        alpha = int(cfg["lora_alpha"])
        scaling = float(alpha) / float(r) if r else None
    except (TypeError, ValueError, ZeroDivisionError) as e:
        raise RuntimeError(f"invalid LoRA scaling in adapter_config.json: {e}") from e
    return {
        "ok": True,
        "sha256": digest,
        "r": r,
        "lora_alpha": alpha,
        "lora_scaling": scaling,
    }


def bind_unit_checkpoint_provenance(
    *,
    family: str,
    seed: int | None,
    subfolder: str,
    base_snapshot_dir: Path | str,
    adapter_snapshot_dir: Path | str,
    expected_sha256_by_relpath: Mapping[str, str],
    require_section11_tokenizer_pins: bool = True,
    require_authoritative_trust: bool = False,
    trust_auth: Any | None = None,
) -> dict[str, Any]:
    """Single fail-closed unit↔checkpoint provenance contract (D5 / D5-A / D5-B).

    Binds the requested family/seed/subfolder to verified local snapshot bytes,
    inventories load-relevant files against an exact pinned inventory,
    authenticates tokenizer/template + adapter_config, and returns **computed**
    digests only. Never invents hashes or falls back to caller-expected digests.

    When ``require_authoritative_trust=True``, ``trust_auth`` must be a
    successful ``CheckpointManifestAuth`` (production_unlocked or
    engineering_only). Arbitrary caller manifests alone are not authoritative.
    """
    from traces.q1.checkpoint_trust import (
        BASE_SAFETENSORS_INDEX_NAME,
        FORBIDDEN_ADAPTER_LOAD_FILES,
        REQUIRED_ADAPTER_LOAD_FILES,
        REQUIRED_BASE_META_FILES,
        parse_safetensors_index,
        required_base_files_from_authenticated_index,
    )

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
    if expected_sf not in required_adapter_subfolders():
        raise RuntimeError(
            f"subfolder {expected_sf!r} is not one of the ten scientific "
            f"adapter identities (D5)"
        )

    if require_authoritative_trust:
        if trust_auth is None or not getattr(trust_auth, "ok", False):
            raise RuntimeError(
                "authoritative checkpoint trust auth required — refuse "
                "caller-supplied manifest as independent proof (D5-A)"
            )
        # production_unlocked OR engineering_only+ok; never silent caller dict
        if not (
            getattr(trust_auth, "production_unlocked", False)
            or getattr(trust_auth, "engineering_only", False)
        ):
            raise RuntimeError(
                "checkpoint trust auth is neither production_unlocked nor "
                "engineering_only — refuse (D5-A)"
            )

    if not expected_sha256_by_relpath:
        raise RuntimeError(
            "authoritative unit weight manifest unavailable — block load without "
            "inventing hashes (D5)"
        )

    # Reject traversal / malformed keys up front
    for key in expected_sha256_by_relpath:
        role_key = str(key)
        if ":" in role_key:
            rel = role_key.split(":", 1)[1]
        else:
            rel = role_key
        try:
            if Path(rel).is_absolute() or ".." in Path(rel).parts:
                raise RuntimeError(f"malformed / traversal manifest key: {key!r}")
        except RuntimeError:
            raise

    snapshot_bind = assert_load_uses_verified_snapshot_bytes(
        base_snapshot_dir=base_snapshot_dir,
        adapter_snapshot_dir=adapter_snapshot_dir,
        expected_sha256_by_relpath=expected_sha256_by_relpath,
        required_adapter_subfolder=subfolder,
    )
    adapter_dir = resolve_unit_adapter_subfolder_dir(adapter_snapshot_dir, subfolder)
    base_root = Path(base_snapshot_dir)
    _reject_symlink(base_root, label="base snapshot")

    weight_inventory = inventory_unit_weight_files(
        base_snapshot_dir=base_snapshot_dir,
        adapter_subfolder_dir=adapter_dir,
    )
    load_inventory = inventory_load_relevant_files(
        base_snapshot_dir=base_snapshot_dir,
        adapter_subfolder_dir=adapter_dir,
    )
    if not weight_inventory["base"] or not weight_inventory["adapter"]:
        raise RuntimeError(
            f"weight inventory empty for subfolder {subfolder!r} (D5)"
        )

    base_exp, ad_exp = _split_manifest_roles(expected_sha256_by_relpath)
    computed = dict(snapshot_bind["computed_sha256"])
    sf_prefix = subfolder.rstrip("/") + "/"

    # Exact inventory: every load-relevant on-disk file must be listed; required
    # names must be present (D5-B / D5-B1 sharded base).
    for name in REQUIRED_BASE_META_FILES:
        if name not in base_exp and not any(Path(k).name == name for k in base_exp):
            raise RuntimeError(
                f"authoritative inventory missing required base file {name} (D5-B1)"
            )
    # Authenticate index bytes then derive exact shard inventory from weight_map
    index_rel = BASE_SAFETENSORS_INDEX_NAME
    if index_rel not in base_exp:
        hits = [k for k in base_exp if Path(k).name == index_rel]
        if len(hits) != 1:
            raise RuntimeError(
                f"base safetensors index not uniquely listed in manifest (D5-B1)"
            )
        index_rel = hits[0]
    index_path = base_root / index_rel
    _reject_symlink(index_path, label="base safetensors index")
    if not index_path.is_file():
        raise RuntimeError(f"missing base safetensors index: {index_path} (D5-B1)")
    index_bytes = index_path.read_bytes()
    index_digest = hashlib.sha256(index_bytes).hexdigest()
    if index_digest != base_exp[index_rel]:
        raise RuntimeError(
            f"base index SHA mismatch: {index_digest} != {base_exp[index_rel]} (D5-B1)"
        )
    required_base_names = required_base_files_from_authenticated_index(
        index_bytes=index_bytes,
        base_files_manifest={
            (f"base:{k}" if not str(k).startswith("base:") else str(k)): v
            for k, v in base_exp.items()
        },
    )
    parsed_index = parse_safetensors_index(index_bytes)
    for name in REQUIRED_ADAPTER_LOAD_FILES:
        rel = f"{subfolder.rstrip('/')}/{name}"
        if rel not in ad_exp:
            raise RuntimeError(
                f"authoritative inventory missing required adapter file "
                f"adapter:{rel} (D5-B)"
            )

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
        raise RuntimeError(
            f"inventoried adapter file not in unit manifest as "
            f"adapter:{rel!r}: {name} (refuse cross-seed / unlisted loadable)"
        )

    disk_base_names = {Path(p).name for p in load_inventory["base"]}
    required_base_set = set(required_base_names)
    missing_base = required_base_set - disk_base_names
    if missing_base:
        raise RuntimeError(
            f"base snapshot missing required sharded files {sorted(missing_base)} "
            "(D5-B1)"
        )
    extra_base = disk_base_names - required_base_set
    # Allow only exact required inventory — no unlisted loadable base files
    if extra_base:
        raise RuntimeError(
            f"unlisted load-relevant base files refused: {sorted(extra_base)} "
            "(D5-B1 exact shard inventory)"
        )
    for abs_path in load_inventory["base"]:
        name = Path(abs_path).name
        matched = _match_base(name)
        digest = hashlib.sha256(Path(abs_path).read_bytes()).hexdigest()
        if digest != base_exp[matched]:
            raise RuntimeError(
                f"base inventory SHA mismatch {abs_path}: "
                f"{digest} != {base_exp[matched]}"
            )
        computed[f"inventory:base:{matched}"] = digest
    computed["base_safetensors_index"] = index_digest
    computed["base_shards"] = ",".join(parsed_index["shards"])

    for abs_path in load_inventory["adapter"]:
        name = Path(abs_path).name
        if name in FORBIDDEN_ADAPTER_LOAD_FILES:
            raise RuntimeError(
                f"competing tokenizer source present: {name} (D5-B)"
            )
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

    # Manifest must not list files absent from disk for this unit's load set
    listed_adapter_names = {Path(k).name for k in ad_exp if k.startswith(sf_prefix)}
    disk_adapter_names = {Path(p).name for p in load_inventory["adapter"]}
    missing_on_disk = listed_adapter_names - disk_adapter_names
    # Allow manifest to list only required load files; extras on disk already fail
    # via _match_adapter. Missing listed required files:
    for name in REQUIRED_ADAPTER_LOAD_FILES:
        if name not in disk_adapter_names:
            raise RuntimeError(
                f"required load file missing on disk: {name} under {subfolder} (D5-B)"
            )
    _ = missing_on_disk  # extras listed but absent: caught by snapshot verify

    tok_pins = (
        FROZEN_SECTION11_TOKENIZER_SHA256
        if require_section11_tokenizer_pins
        else {
            name: ad_exp[f"{subfolder.rstrip('/')}/{name}"]
            for name in REQUIRED_TOKENIZER_TEMPLATE_FILES
            if f"{subfolder.rstrip('/')}/{name}" in ad_exp
        }
    )
    tok = require_tokenizer_and_template_files(
        adapter_dir,
        expected_sha256=tok_pins if tok_pins else None,
    )
    for name, digest in tok["files"].items():
        computed[f"tokenizer:{name}"] = digest

    adapter_cfg = _verify_adapter_config_scaling(adapter_dir, computed=computed)
    # Cross-check adapter_config digest against authoritative inventory
    cfg_rel = f"{subfolder.rstrip('/')}/adapter_config.json"
    if cfg_rel in ad_exp and adapter_cfg["sha256"] != ad_exp[cfg_rel]:
        raise RuntimeError(
            f"adapter_config.json digest mismatch vs inventory (D5-B): "
            f"{adapter_cfg['sha256']} != {ad_exp[cfg_rel]}"
        )

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
        "inventory": weight_inventory,
        "load_relevant_inventory": load_inventory,
        "local_paths": local_paths,
        "tokenizer_template_files": tok,
        "adapter_config": adapter_cfg,
        "base_shards": list(parsed_index["shards"]),
        "base_weight_map_n": len(parsed_index["weight_map"]),
        "computed_sha256": computed,
        "snapshot_byte_binding": snapshot_bind,
        "trust": {
            "require_authoritative_trust": require_authoritative_trust,
            "production_unlocked": bool(
                getattr(trust_auth, "production_unlocked", False)
            )
            if trust_auth is not None
            else False,
            "engineering_only": bool(
                getattr(trust_auth, "engineering_only", False)
            )
            if trust_auth is not None
            else False,
        },
    }


def engineering_bind_unit_checkpoint_provenance(
    *,
    family: str,
    seed: int | None,
    subfolder: str,
    base_snapshot_dir: Path | str,
    adapter_snapshot_dir: Path | str,
    engineering_manifest: Mapping[str, Any],
    require_section11_tokenizer_pins: bool = False,
) -> dict[str, Any]:
    """Isolated engineering-test bind — never production authorization (D5-A)."""
    from traces.q1.checkpoint_trust import (
        authenticate_engineering_checkpoint_manifest,
        unit_digests_from_auth,
    )

    auth = authenticate_engineering_checkpoint_manifest(engineering_manifest)
    if not auth.ok:
        raise RuntimeError(
            "engineering checkpoint manifest auth failed: "
            + "; ".join(auth.failures)
        )
    digests = unit_digests_from_auth(auth, subfolder=subfolder)
    return bind_unit_checkpoint_provenance(
        family=family,
        seed=seed,
        subfolder=subfolder,
        base_snapshot_dir=base_snapshot_dir,
        adapter_snapshot_dir=adapter_snapshot_dir,
        expected_sha256_by_relpath=digests,
        require_section11_tokenizer_pins=require_section11_tokenizer_pins,
        require_authoritative_trust=True,
        trust_auth=auth,
    )
