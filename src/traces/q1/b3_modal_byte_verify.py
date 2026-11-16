"""B3 Modal CPU checkpoint-byte verification helpers (no model load / no GPU).

Phase: prepare exact-revision download + independent SHA-256 inventory +
candidate manifest. Scientific authorization remains OFF. Production pin stays
unset. Billable downloads require ``B3_OWNER_APPROVED_MAX_USD`` > 0 reflecting
an owner hard cap confirmed **in chat**.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Pinned identities (duplicated here to avoid importing model_loader → torch).
BASE_MODEL_ID = "google/gemma-2-2b"
DENSE_ADAPTER_REPO = "interpretable-finetuning/topklora-gemma-2-2b-dense-lora"
PINNED_BASE_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
PINNED_ADAPTER_REVISION = "6303a74fc832a34d584999364817268aefd9623b"

from traces.q1.checkpoint_provenance import FROZEN_SECTION11_TOKENIZER_SHA256
from traces.q1.checkpoint_trust import (
    ADAPTER_WEIGHT_BASENAME,
    BASE_SAFETENSORS_INDEX_NAME,
    CHECKPOINT_MANIFEST_V1,
    FORBIDDEN_ADAPTER_LOAD_FILES,
    PINNED_CHECKPOINT_MANIFEST_SHA256,
    REQUIRED_ADAPTER_LOAD_FILES,
    REQUIRED_BASE_META_FILES,
    canonical_checkpoint_manifest_bytes,
    parse_safetensors_index,
    required_adapter_subfolders,
    sha256_bytes,
    validate_checkpoint_manifest_schema,
)

PROGRAM_NAME = "q1_b3_modal_byte_verify"
PROGRAM_VERSION = "0.2.0"

MODAL_APP_NAME = "traces-q1-b3-byte-verify"
MODAL_VOLUME_NAME = "traces-q1-b3-checkpoint-bytes"
# Existing owner Modal secret (keys expected: HF_TOKEN or HUGGING_FACE_HUB_TOKEN).
MODAL_SECRET_NAME = "hf-token"
MODAL_MOUNT_PATH = "/mnt/b3"

OWNER_BUDGET_ENV = "B3_OWNER_APPROVED_MAX_USD"
BILLABLE_CONFIRM_ENV = "B3_CHAT_CAP_CONFIRMED"
OWNER_APPROVED_CAP_USD = 10.0  # chat-confirmed 2026-11-16 hard ceiling

# Advertised Phase-0 minimum (metadata only; not a local hash).
ADVERTISED_MIN_TRANSFER_BYTES = 11_139_405_966

# Software cost controls (Starter plan has no environment budget API).
MODAL_CPU_CORES = 2.0
MODAL_MEMORY_GIB = 8.0
MODAL_TIMEOUT_SEC = 45 * 60  # hard stop
MODAL_CPU_USD_PER_CORE_HOUR = 0.0473
MODAL_MEM_USD_PER_GIB_HOUR = 0.008
# Conservative preflight estimate ceiling must stay under owner cap.
COST_ESTIMATE_SAFETY_FACTOR = 2.0

CHUNK_SIZE = 1024 * 1024  # 1 MiB — never read multi-GiB shards wholesale

# Files never copied into the trusted materialized adapter inventory.
ADAPTER_EXCLUDE_BASENAMES: frozenset[str] = frozenset(
    {
        "sleeper_run_config.json",
        "topk_config.json",
        "tokenizer.json",
        "tokenizer.model.json",
        "README.md",
        ".gitattributes",
    }
) | FORBIDDEN_ADAPTER_LOAD_FILES


class B3VerifyError(RuntimeError):
    """Fail-closed B3 verification error."""


@dataclass
class FileDigestRecord:
    relative_path: str
    role: str  # base | adapter
    sha256: str
    size_bytes: int
    source_repo: str
    source_revision: str
    source_path: str


@dataclass
class B3AuditReport:
    ok: bool
    mode: str
    program: str = PROGRAM_NAME
    program_version: str = PROGRAM_VERSION
    timestamp_utc: str = ""
    base_repo: str = BASE_MODEL_ID
    base_revision: str = PINNED_BASE_REVISION
    adapter_repo: str = DENSE_ADAPTER_REPO
    adapter_revision: str = PINNED_ADAPTER_REVISION
    production_pin: str | None = None
    failures: list[str] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    section11: dict[str, Any] = field(default_factory=dict)
    adapter_weight_distinct: bool | None = None
    candidate_manifest_sha256: str | None = None
    candidate_manifest_path: str | None = None
    notes: list[str] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def expected_ten_units() -> tuple[str, ...]:
    return required_adapter_subfolders()


def require_production_pin_unset() -> None:
    if PINNED_CHECKPOINT_MANIFEST_SHA256 is not None:
        raise B3VerifyError(
            "PINNED_CHECKPOINT_MANIFEST_SHA256 is set — B3 tooling must not run "
            "while a production pin is already frozen without a separate owner "
            f"instrument (got {PINNED_CHECKPOINT_MANIFEST_SHA256!r})"
        )


def parse_owner_budget_usd(raw: str | None) -> float:
    """Fail closed on missing / zero / non-positive / non-numeric budget."""
    if raw is None or not str(raw).strip():
        raise B3VerifyError(
            f"{OWNER_BUDGET_ENV} is missing — owner must set a hard USD cap in chat "
            "before billable B3 downloads"
        )
    text = str(raw).strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise B3VerifyError(
            f"{OWNER_BUDGET_ENV}={text!r} is not a valid USD amount"
        ) from exc
    if not (value > 0.0) or value != value:  # NaN check
        raise B3VerifyError(
            f"{OWNER_BUDGET_ENV}={text!r} must be a finite number > 0"
        )
    return value


def estimate_modal_cpu_job_usd(
    *,
    timeout_sec: float = MODAL_TIMEOUT_SEC,
    cpu_cores: float = MODAL_CPU_CORES,
    memory_gib: float = MODAL_MEMORY_GIB,
) -> dict[str, float]:
    """Upper-bound Modal CPU/memory cost if the job runs to timeout."""
    hours = float(timeout_sec) / 3600.0
    cpu = hours * cpu_cores * MODAL_CPU_USD_PER_CORE_HOUR
    mem = hours * memory_gib * MODAL_MEM_USD_PER_GIB_HOUR
    raw = cpu + mem
    return {
        "cpu_usd": round(cpu, 4),
        "memory_usd": round(mem, 4),
        "timeout_bound_usd": round(raw, 4),
        "safety_bound_usd": round(raw * COST_ESTIMATE_SAFETY_FACTOR, 4),
    }


def require_billable_gates(*, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Gates for any paid Modal download / hashing job."""
    e = dict(os.environ if env is None else env)
    require_production_pin_unset()
    budget = parse_owner_budget_usd(e.get(OWNER_BUDGET_ENV))
    confirm = (e.get(BILLABLE_CONFIRM_ENV) or "").strip().lower()
    if confirm not in {"1", "true", "yes"}:
        raise B3VerifyError(
            f"{BILLABLE_CONFIRM_ENV} must be set to 1/true/yes after the owner "
            "states the hard B3 USD cap in chat (currently unset/false)"
        )
    if abs(budget - OWNER_APPROVED_CAP_USD) > 1e-9 and budget > OWNER_APPROVED_CAP_USD:
        raise B3VerifyError(
            f"{OWNER_BUDGET_ENV}={budget} exceeds chat-approved hard cap "
            f"${OWNER_APPROVED_CAP_USD}"
        )
    # Require the env value to equal the chat-approved $10 (exact match).
    if abs(budget - OWNER_APPROVED_CAP_USD) > 1e-9:
        raise B3VerifyError(
            f"{OWNER_BUDGET_ENV} must equal the chat-approved hard cap "
            f"{OWNER_APPROVED_CAP_USD} (got {budget})"
        )
    est = estimate_modal_cpu_job_usd()
    if est["safety_bound_usd"] > budget:
        raise B3VerifyError(
            f"cannot reliably keep Modal CPU job under ${budget}: "
            f"timeout safety-bound estimate ${est['safety_bound_usd']} "
            f"(timeout={MODAL_TIMEOUT_SEC}s, cpu={MODAL_CPU_CORES}, "
            f"mem={MODAL_MEMORY_GIB}GiB). Reduce resources/timeout or raise cap."
        )
    return {
        "owner_approved_max_usd": budget,
        "chat_cap_confirmed": True,
        "advertised_min_transfer_bytes": ADVERTISED_MIN_TRANSFER_BYTES,
        "cost_estimate": est,
        "controls": {
            "cpu_cores": MODAL_CPU_CORES,
            "memory_gib": MODAL_MEMORY_GIB,
            "timeout_sec": MODAL_TIMEOUT_SEC,
            "gpu": None,
            "max_containers": 1,
            "volume_name": MODAL_VOLUME_NAME,
            "secret_name": MODAL_SECRET_NAME,
        },
    }


def _lexically_under(path: Path, root: Path) -> bool:
    """True iff ``path`` is lexically equal to or under ``root`` (no resolve)."""
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def assert_path_inside_trust_root(path: Path, within: Path) -> None:
    """Fail closed if lexical or resolved path escapes the materialized trust root.

    Walks **lexical** components from ``within`` toward ``path`` using ``lstat`` /
    ``is_symlink`` on every component (files and directories). Modal Volume mount
    prefixes outside ``within`` are irrelevant; everything under the declared
    materialized root must be symlink-free and must not escape via resolve().
    """
    path = Path(path)
    within = Path(within)
    if not _lexically_under(path, within):
        raise B3VerifyError(
            f"path escapes trust root lexically: {path} not under {within}"
        )
    # Walk lexical components; refuse any symlink (dir or file) on the chain.
    rel = path.relative_to(within)
    cur = within
    # The trust root itself must exist as a real directory (not a symlink).
    if within.is_symlink():
        raise B3VerifyError(f"trust root itself is a symlink: {within}")
    if not within.is_dir():
        raise B3VerifyError(f"trust root is not a directory: {within}")
    for part in rel.parts:
        cur = cur / part
        if cur.is_symlink():
            raise B3VerifyError(f"symlink in trust path refused: {cur}")
        if not cur.exists():
            raise B3VerifyError(f"missing path component under trust root: {cur}")
    # Resolved containment — never silently ignore relative_to failures.
    try:
        resolved_path = path.resolve(strict=True)
        resolved_root = within.resolve(strict=True)
    except FileNotFoundError as exc:
        raise B3VerifyError(f"resolve failed under trust root: {exc}") from exc
    if not _lexically_under(resolved_path, resolved_root):
        raise B3VerifyError(
            f"resolved path escapes trust root: {resolved_path} not under "
            f"{resolved_root} (lexical {path})"
        )


def assert_regular_file_no_symlink(
    path: Path, *, within: Path | None = None
) -> None:
    """Refuse symlink files; when ``within`` is set, enforce full trust-root walk."""
    path = Path(path)
    if within is not None:
        assert_path_inside_trust_root(path, Path(within))
    elif path.is_symlink():
        raise B3VerifyError(f"symlink refused as trusted input: {path}")
    if path.is_symlink():
        raise B3VerifyError(f"symlink refused as trusted input: {path}")
    if not path.is_file():
        raise B3VerifyError(f"not a regular file: {path}")


def assert_dir_no_symlink(path: Path, *, within: Path | None = None) -> None:
    """Refuse symlinked or non-directory entries (unit/base roots)."""
    path = Path(path)
    if within is not None:
        assert_path_inside_trust_root(path, Path(within))
    if path.is_symlink():
        raise B3VerifyError(f"symlinked directory refused: {path}")
    if not path.is_dir():
        raise B3VerifyError(f"not a directory: {path}")


def reject_unsafe_relpath(rel: str) -> str:
    if not rel or not str(rel).strip():
        raise B3VerifyError("empty relative path")
    p = Path(rel)
    if p.is_absolute():
        raise B3VerifyError(f"absolute path forbidden: {rel!r}")
    if ".." in p.parts:
        raise B3VerifyError(f"path traversal forbidden: {rel!r}")
    return str(p.as_posix())


def sha256_file_chunked(
    path: Path,
    *,
    chunk_size: int = CHUNK_SIZE,
    within: Path | None = None,
) -> tuple[str, int]:
    assert_regular_file_no_symlink(path, within=within)
    h = hashlib.sha256()
    total = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    st_size = path.stat().st_size
    if total != st_size:
        raise B3VerifyError(
            f"size mismatch while hashing {path}: read={total} stat={st_size}"
        )
    return h.hexdigest(), total


def git_blob_sha1_file_chunked(
    path: Path, *, chunk_size: int = CHUNK_SIZE, within: Path | None = None
) -> tuple[str, int]:
    """Git blob OID: SHA1(b'blob ' + ascii(size) + b'\\0' + content), streaming."""
    assert_regular_file_no_symlink(path, within=within)
    size = path.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode("ascii"))
    total = 0
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            total += len(chunk)
    if total != size:
        raise B3VerifyError(
            f"size mismatch while git-blob hashing {path}: read={total} stat={size}"
        )
    return h.hexdigest(), total


def materialize_copy_no_symlink(src: Path, dest: Path) -> None:
    """Copy real file bytes into dest; never leave dest as a symlink."""
    real = Path(src).resolve()
    if not real.is_file():
        raise B3VerifyError(f"source is not a regular resolvable file: {src}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    # Byte-copy via open() so dest is a fresh regular file on the Volume.
    with real.open("rb") as inf, dest.open("wb") as outf:
        while True:
            chunk = inf.read(CHUNK_SIZE)
            if not chunk:
                break
            outf.write(chunk)
    if dest.is_symlink() or not dest.is_file():
        raise B3VerifyError(f"materialized dest is not a regular file: {dest}")


def parse_index_shards(index_path: Path, *, within: Path | None = None) -> list[str]:
    assert_regular_file_no_symlink(index_path, within=within)
    raw = index_path.read_bytes()
    parsed = parse_safetensors_index(raw)
    return list(parsed["shards"])


def verify_section11_unit(
    unit_dir: Path, *, within: Path | None = None
) -> dict[str, Any]:
    """Compare each of four §11 files to frozen pins using local SHA-256."""
    trust = within if within is not None else unit_dir
    results: dict[str, Any] = {}
    failures: list[str] = []
    for name, expected in FROZEN_SECTION11_TOKENIZER_SHA256.items():
        path = unit_dir / name
        if not path.exists():
            failures.append(f"missing §11 file {name}")
            results[name] = {"ok": False, "reason": "missing"}
            continue
        try:
            digest, size = sha256_file_chunked(path, within=trust)
        except B3VerifyError as exc:
            failures.append(str(exc))
            results[name] = {"ok": False, "reason": str(exc)}
            continue
        ok = digest == expected
        results[name] = {
            "ok": ok,
            "sha256": digest,
            "expected": expected,
            "size_bytes": size,
        }
        if not ok:
            failures.append(f"§11 mismatch {name}: got {digest} expected {expected}")
    for banned in FORBIDDEN_ADAPTER_LOAD_FILES:
        banned_path = unit_dir / banned
        if banned_path.exists() or banned_path.is_symlink():
            failures.append(f"forbidden competing tokenizer present: {banned}")
    return {"ok": not failures, "files": results, "failures": failures}


def verify_adapter_unit_inventory(
    unit_dir: Path, *, subfolder: str, adapter_root: Path | None = None
) -> dict[str, Any]:
    reject_unsafe_relpath(subfolder)
    if subfolder not in expected_ten_units():
        raise B3VerifyError(f"unexpected adapter subfolder {subfolder!r}")
    trust = adapter_root if adapter_root is not None else unit_dir
    try:
        assert_dir_no_symlink(unit_dir, within=trust)
    except B3VerifyError as exc:
        return {
            "ok": False,
            "subfolder": subfolder,
            "digests": {},
            "sizes": {},
            "section11": {},
            "failures": [str(exc)],
        }
    present: dict[str, Path] = {}
    failures: list[str] = []
    for p in unit_dir.iterdir():
        if p.name.startswith("."):
            continue
        if p.is_symlink():
            failures.append(f"{subfolder}: symlink entry refused: {p.name}")
            continue
        if p.is_dir():
            failures.append(f"{subfolder}: unexpected directory: {p.name}")
            continue
        if p.is_file():
            present[p.name] = p
        else:
            failures.append(f"{subfolder}: non-file entry: {p.name}")
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for name in REQUIRED_ADAPTER_LOAD_FILES:
        if name not in present:
            failures.append(f"{subfolder}: missing required {name}")
            continue
        try:
            digest, size = sha256_file_chunked(present[name], within=trust)
        except B3VerifyError as exc:
            failures.append(str(exc))
            continue
        digests[name] = digest
        sizes[name] = size
    for name in present:
        if name in FORBIDDEN_ADAPTER_LOAD_FILES:
            failures.append(f"{subfolder}: forbidden file present {name}")
        elif name not in REQUIRED_ADAPTER_LOAD_FILES:
            failures.append(
                f"{subfolder}: unexpected extra file in materialized unit dir: {name}"
            )
    sec11 = verify_section11_unit(unit_dir, within=trust)
    failures.extend(sec11["failures"])
    return {
        "ok": not failures,
        "subfolder": subfolder,
        "digests": digests,
        "sizes": sizes,
        "section11": sec11,
        "failures": failures,
    }


def assert_adapter_weights_pairwise_distinct(
    weight_digests: Mapping[str, str],
) -> None:
    seen: dict[str, str] = {}
    for sf, digest in weight_digests.items():
        if digest in seen:
            raise B3VerifyError(
                f"adapter-weight digests not pairwise distinct: {sf} collides "
                f"with {seen[digest]}"
            )
        seen[digest] = sf
    if len(weight_digests) != 10 or len(seen) != 10:
        raise B3VerifyError(
            f"expected 10 distinct adapter-weight digests, got "
            f"{len(weight_digests)} entries / {len(seen)} unique"
        )


def build_candidate_manifest(
    *,
    base_digests: Mapping[str, str],
    unit_digests: Mapping[str, Mapping[str, str]],
    notes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build schema-valid candidate manifest (never production path / pin)."""
    units: dict[str, Any] = {}
    for sf in expected_ten_units():
        if sf not in unit_digests:
            raise B3VerifyError(f"candidate missing unit {sf}")
        family, seed_tok = sf.split("/", 1)
        files = {
            f"adapter:{sf}/{name}": digest
            for name, digest in unit_digests[sf].items()
        }
        units[sf] = {
            "family": family,
            "seed": int(seed_tok.removeprefix("seed")),
            "subfolder": sf,
            "files": files,
        }
    base_files = {
        (k if str(k).startswith("base:") else f"base:{k}"): v
        for k, v in base_digests.items()
    }
    payload = {
        "artifact_type": CHECKPOINT_MANIFEST_V1,
        "self_generated": False,
        "base_files": base_files,
        "units": units,
        "notes": list(notes or [])
        + [
            "B3_CANDIDATE_ONLY — not production authorization",
            "PINNED_CHECKPOINT_MANIFEST_SHA256 must remain unset until owner freeze",
        ],
    }
    failures = validate_checkpoint_manifest_schema(payload)
    if failures:
        raise B3VerifyError(
            "candidate manifest schema failures: " + "; ".join(failures)
        )
    return payload


def inventory_materialized_snapshot(
    *,
    base_dir: Path,
    adapter_root: Path,
) -> tuple[B3AuditReport, dict[str, Any] | None]:
    """Hash a local no-symlink snapshot and build a candidate manifest."""
    require_production_pin_unset()
    failures: list[str] = []
    file_rows: list[dict[str, Any]] = []
    base_digests: dict[str, str] = {}

    try:
        assert_dir_no_symlink(base_dir)
        assert_dir_no_symlink(adapter_root)
    except B3VerifyError as exc:
        return (
            B3AuditReport(
                ok=False,
                mode="inventory",
                timestamp_utc=utc_now(),
                production_pin=PINNED_CHECKPOINT_MANIFEST_SHA256,
                failures=[str(exc)],
            ),
            None,
        )

    index_path = base_dir / BASE_SAFETENSORS_INDEX_NAME
    try:
        shards = parse_index_shards(index_path, within=base_dir)
    except Exception as exc:  # noqa: BLE001 — surface as audit failure
        return (
            B3AuditReport(
                ok=False,
                mode="inventory",
                timestamp_utc=utc_now(),
                production_pin=PINNED_CHECKPOINT_MANIFEST_SHA256,
                failures=[f"index parse failed: {exc}"],
            ),
            None,
        )

    required_base = list(REQUIRED_BASE_META_FILES) + shards
    required_base_set = set(required_base)
    # Materialized trust root must contain only inventory files (no tokenizer.json etc.)
    for path in sorted(base_dir.iterdir()):
        if path.name.startswith(".") or path.name == ".hf_staging":
            continue
        if path.is_symlink():
            failures.append(f"base symlink entry refused: {path.name}")
            continue
        if path.is_dir():
            failures.append(f"unexpected directory in base materialization: {path.name}")
            continue
        if path.is_file():
            if path.name not in required_base_set:
                failures.append(
                    f"base contains non-inventory file "
                    f"(must not be in trust root): {path.name}"
                )
        else:
            failures.append(f"base non-file entry refused: {path.name}")

    for name in required_base:
        reject_unsafe_relpath(name)
        path = base_dir / name
        try:
            digest, size = sha256_file_chunked(path, within=base_dir)
        except B3VerifyError as exc:
            failures.append(str(exc))
            continue
        key = f"base:{name}"
        base_digests[key] = digest
        file_rows.append(
            {
                "relative_path": name,
                "role": "base",
                "sha256": digest,
                "size_bytes": size,
                "source_repo": BASE_MODEL_ID,
                "source_revision": PINNED_BASE_REVISION,
                "source_path": name,
            }
        )

    unit_digests: dict[str, dict[str, str]] = {}
    weight_digests: dict[str, str] = {}
    section11_by_unit: dict[str, Any] = {}
    for sf in expected_ten_units():
        unit_dir = adapter_root.joinpath(*sf.split("/"))
        if unit_dir.is_symlink() or not unit_dir.is_dir():
            failures.append(f"missing or symlinked unit directory {sf}")
            continue
        inv = verify_adapter_unit_inventory(
            unit_dir, subfolder=sf, adapter_root=adapter_root
        )
        section11_by_unit[sf] = inv["section11"]
        if not inv["ok"]:
            failures.extend(inv["failures"])
        unit_digests[sf] = dict(inv["digests"])
        if ADAPTER_WEIGHT_BASENAME in inv["digests"]:
            weight_digests[sf] = inv["digests"][ADAPTER_WEIGHT_BASENAME]
        for name, digest in inv["digests"].items():
            file_rows.append(
                {
                    "relative_path": f"{sf}/{name}",
                    "role": "adapter",
                    "sha256": digest,
                    "size_bytes": inv["sizes"].get(name),
                    "source_repo": DENSE_ADAPTER_REPO,
                    "source_revision": PINNED_ADAPTER_REVISION,
                    "source_path": f"{sf}/{name}",
                }
            )

    distinct_ok: bool | None = None
    try:
        if len(weight_digests) == 10:
            assert_adapter_weights_pairwise_distinct(weight_digests)
            distinct_ok = True
        else:
            distinct_ok = False
            failures.append("incomplete adapter-weight digest set")
    except B3VerifyError as exc:
        distinct_ok = False
        failures.append(str(exc))

    candidate: dict[str, Any] | None = None
    candidate_sha = None
    if not failures and len(unit_digests) == 10:
        try:
            candidate = build_candidate_manifest(
                base_digests=base_digests,
                unit_digests=unit_digests,
            )
            candidate_sha = sha256_bytes(canonical_checkpoint_manifest_bytes(candidate))
        except B3VerifyError as exc:
            failures.append(str(exc))
            candidate = None

    report = B3AuditReport(
        ok=not failures,
        mode="inventory",
        timestamp_utc=utc_now(),
        production_pin=PINNED_CHECKPOINT_MANIFEST_SHA256,
        failures=failures,
        files=file_rows,
        section11=section11_by_unit,
        adapter_weight_distinct=distinct_ok,
        candidate_manifest_sha256=candidate_sha,
        notes=[
            "candidate only — not CHECKPOINT_PROVENANCE_MANIFEST.json",
            "downloader self-check is not independent Agent 1 audit",
        ],
    )
    return report, candidate


def write_candidate_artifacts(
    report: B3AuditReport,
    candidate: Mapping[str, Any] | None,
    out_dir: Path,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Never write production path.
    prod = out_dir / "CHECKPOINT_PROVENANCE_MANIFEST.json"
    if prod.exists():
        raise B3VerifyError(
            "refusing to write near existing production manifest filename in out_dir"
        )
    report_path = out_dir / "B3_CANDIDATE_AUDIT_REPORT.json"
    payload = report.to_dict()
    # Redact anything that looks like a token
    text = json.dumps(payload, indent=2, sort_keys=True)
    text = re.sub(r"(hf_[A-Za-z0-9]+)", "[REDACTED_HF_TOKEN]", text)
    report_path.write_text(text + "\n", encoding="utf-8")
    paths = {"audit_report": str(report_path)}
    if candidate is not None and report.ok:
        man_path = out_dir / "B3_CANDIDATE_CHECKPOINT_MANIFEST.json"
        man_path.write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths["candidate_manifest"] = str(man_path)
        report.candidate_manifest_path = str(man_path)
    return paths


def preflight_local(*, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """No-download preflight: constants, pin unset, budget gate status."""
    e = dict(os.environ if env is None else env)
    require_production_pin_unset()
    budget_status: dict[str, Any]
    try:
        budget = parse_owner_budget_usd(e.get(OWNER_BUDGET_ENV))
        budget_status = {"present": True, "value_usd": budget}
    except B3VerifyError as exc:
        budget_status = {"present": False, "reason": str(exc)}

    confirm = (e.get(BILLABLE_CONFIRM_ENV) or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    billable_ready = bool(budget_status.get("present")) and confirm
    return {
        "ok": True,
        "mode": "preflight",
        "program": PROGRAM_NAME,
        "program_version": PROGRAM_VERSION,
        "timestamp_utc": utc_now(),
        "base_repo": BASE_MODEL_ID,
        "base_revision": PINNED_BASE_REVISION,
        "adapter_repo": DENSE_ADAPTER_REPO,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "ten_units": list(expected_ten_units()),
        "required_adapter_files": list(REQUIRED_ADAPTER_LOAD_FILES),
        "required_base_meta": list(REQUIRED_BASE_META_FILES),
        "forbidden_adapter_files": sorted(FORBIDDEN_ADAPTER_LOAD_FILES),
        "modal": {
            "app_name": MODAL_APP_NAME,
            "volume_name": MODAL_VOLUME_NAME,
            "secret_name": MODAL_SECRET_NAME,
            "mount_path": MODAL_MOUNT_PATH,
            "cpu_only": True,
            "gpu": None,
            "concurrency": 1,
        },
        "production_pin": PINNED_CHECKPOINT_MANIFEST_SHA256,
        "advertised_min_transfer_bytes": ADVERTISED_MIN_TRANSFER_BYTES,
        "budget": budget_status,
        "chat_cap_confirmed": confirm,
        "billable_download_permitted": billable_ready,
        "notes": [
            "Default mode does not download bytes or launch paid Modal jobs",
            "Set B3_OWNER_APPROVED_MAX_USD and B3_CHAT_CAP_CONFIRMED=1 after "
            "owner states the hard cap in chat before --mode download",
            "Owner confirmed Gemma access + Modal venue in chat; that is not a USD cap",
        ],
    }


def assert_no_model_load_imports(stack_modules: Iterable[str]) -> None:
    """Guard used in tests: refuse accidental transformers model load modules."""
    banned = {
        "transformers.models",
        "peft.peft_model",
        "peft.auto",
    }
    for name in stack_modules:
        for b in banned:
            if name.startswith(b):
                raise B3VerifyError(f"model-load module imported during B3 verify: {name}")


# --- HF download helpers (only called from billable path) --------------------


def planned_download_paths(shards: Sequence[str]) -> dict[str, list[str]]:
    return {
        "base": list(REQUIRED_BASE_META_FILES) + list(shards),
        "adapters": [
            f"{sf}/{name}"
            for sf in expected_ten_units()
            for name in REQUIRED_ADAPTER_LOAD_FILES
        ],
    }


def resolve_hf_token(env: Mapping[str, str] | None = None) -> str:
    e = dict(os.environ if env is None else env)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        val = (e.get(key) or "").strip()
        if val:
            return val
    raise B3VerifyError(
        "HF token env missing (expected HF_TOKEN or HUGGING_FACE_HUB_TOKEN in "
        f"Modal secret {MODAL_SECRET_NAME!r})"
    )


@dataclass
class SourceObjectIdentity:
    repo_id: str
    revision: str
    path: str
    size: int | None
    kind: str  # lfs_sha256 | git_blob_sha1
    digest: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fetch_source_object_identity(
    *,
    repo_id: str,
    revision: str,
    path: str,
    token: str | None,
    paths_info: Sequence[Any] | None = None,
) -> SourceObjectIdentity:
    """Bind a path to the Hub revision's published content identity.

    LFS → ``lfs.sha256`` vs local streaming SHA-256.
    Non-LFS → ``blob_id`` vs local Git blob OID SHA-1(prefix+body).
    Missing digests → SOURCE_BINDING_BLOCKED (not a size/name match).
    """
    reject_unsafe_relpath(path)
    if paths_info is None:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        paths_info = api.get_paths_info(repo_id, [path], revision=revision)
    if not paths_info:
        raise B3VerifyError(
            f"SOURCE_BINDING_BLOCKED: Hub returned no paths_info for "
            f"{repo_id}@{revision}:{path}"
        )
    info = paths_info[0]
    info_path = getattr(info, "path", None)
    if info_path is not None and str(info_path) != path:
        raise B3VerifyError(
            f"SOURCE_BINDING_BLOCKED: paths_info path mismatch "
            f"(wanted {path!r}, got {info_path!r})"
        )
    # RepoFolder / directory entries have tree_id and no usable content digest.
    if getattr(info, "tree_id", None) is not None and getattr(info, "blob_id", None) is None:
        raise B3VerifyError(
            f"SOURCE_BINDING_BLOCKED: path is a directory/tree, not a file: "
            f"{repo_id}@{revision}:{path}"
        )
    size = getattr(info, "size", None)
    lfs = getattr(info, "lfs", None)
    if lfs is not None:
        sha = getattr(lfs, "sha256", None) or (
            lfs.get("sha256") if isinstance(lfs, Mapping) else None
        )
        if not sha or not isinstance(sha, str) or len(sha) != 64:
            raise B3VerifyError(
                f"SOURCE_BINDING_BLOCKED: LFS object missing sha256 for "
                f"{repo_id}@{revision}:{path}"
            )
        return SourceObjectIdentity(
            repo_id=repo_id,
            revision=revision,
            path=path,
            size=int(size) if size is not None else None,
            kind="lfs_sha256",
            digest=sha.lower(),
        )
    blob_id = getattr(info, "blob_id", None) or getattr(info, "oid", None)
    if not blob_id or not isinstance(blob_id, str) or len(blob_id) != 40:
        raise B3VerifyError(
            f"SOURCE_BINDING_BLOCKED: non-LFS blob_id missing for "
            f"{repo_id}@{revision}:{path}"
        )
    return SourceObjectIdentity(
        repo_id=repo_id,
        revision=revision,
        path=path,
        size=int(size) if size is not None else None,
        kind="git_blob_sha1",
        digest=str(blob_id).lower(),
    )


def verify_local_matches_source(
    path: Path,
    identity: SourceObjectIdentity,
    *,
    within: Path | None = None,
) -> dict[str, Any]:
    """Compare local bytes to source content identity (in addition to path checks)."""
    assert_regular_file_no_symlink(path, within=within)
    local_size = path.stat().st_size
    if identity.size is not None and local_size != identity.size:
        raise B3VerifyError(
            f"source size mismatch for {identity.path}: local={local_size} "
            f"source={identity.size}"
        )
    if identity.kind == "lfs_sha256":
        local_digest, _ = sha256_file_chunked(path, within=within)
    elif identity.kind == "git_blob_sha1":
        local_digest, _ = git_blob_sha1_file_chunked(path, within=within)
    else:
        raise B3VerifyError(f"SOURCE_BINDING_BLOCKED: unknown kind {identity.kind}")
    if local_digest.lower() != identity.digest.lower():
        raise B3VerifyError(
            f"source content identity mismatch for {identity.path}: "
            f"local_{identity.kind}={local_digest} source={identity.digest}"
        )
    return {
        "ok": True,
        "local_digest": local_digest.lower(),
        "source": identity.to_dict(),
        "size_bytes": local_size,
    }


def download_hf_file(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    dest: Path,
    token: str | None,
) -> None:
    """Single-file pinned revision download into dest (no from_pretrained)."""
    reject_unsafe_relpath(filename)
    from huggingface_hub import hf_hub_download

    if not token:
        raise B3VerifyError("HF token missing for gated/authenticated download")
    staging_parent = dest.parent / ".hf_staging"
    staging_parent.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "filename": filename,
        "revision": revision,
        "token": token,
        "local_dir": str(staging_parent),
        "force_download": False,
    }
    try:
        local = hf_hub_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        local = hf_hub_download(**kwargs)
    local_path = Path(local)
    if local_path.is_symlink():
        local_path = local_path.resolve()
    materialize_copy_no_symlink(local_path, dest)
    try:
        if staging_parent.is_dir():
            shutil.rmtree(staging_parent, ignore_errors=True)
    except OSError:
        pass


def ensure_materialized_file(
    *,
    repo_id: str,
    revision: str,
    filename: str,
    dest: Path,
    token: str,
    within: Path,
    allow_download: bool = True,
    source_identity: SourceObjectIdentity | None = None,
) -> dict[str, Any]:
    """Reuse dest only after symlink + source-identity checks; else download.

    A bare ``dest.exists()`` cache hit is never trusted.
    """
    reject_unsafe_relpath(filename)
    identity = source_identity or fetch_source_object_identity(
        repo_id=repo_id, revision=revision, path=filename, token=token
    )
    if dest.exists() or dest.is_symlink():
        binding = verify_local_matches_source(dest, identity, within=within)
        return {"action": "reused_after_source_bind", **binding}
    if not allow_download:
        raise B3VerifyError(
            f"missing materialized file and download disabled: {dest}"
        )
    download_hf_file(
        repo_id=repo_id,
        revision=revision,
        filename=filename,
        dest=dest,
        token=token,
    )
    binding = verify_local_matches_source(dest, identity, within=within)
    return {"action": "downloaded_and_source_bound", **binding}


def validate_existing_materialized_source_bound(
    *,
    root: Path,
    token: str,
) -> dict[str, Any]:
    """Bind every already-materialized inventory file to source OIDs (no download).

    Refuses reuse of ``dest.exists()`` without source identity + no-symlink checks.
    Does not overwrite Volume bytes.
    """
    require_production_pin_unset()
    base_dir = root / "base" / PINNED_BASE_REVISION / "materialized"
    adapter_root = root / "adapters" / PINNED_ADAPTER_REVISION / "materialized"
    assert_dir_no_symlink(base_dir)
    assert_dir_no_symlink(adapter_root)

    bindings: list[dict[str, Any]] = []
    failures: list[str] = []

    index_dest = base_dir / BASE_SAFETENSORS_INDEX_NAME
    try:
        bindings.append(
            ensure_materialized_file(
                repo_id=BASE_MODEL_ID,
                revision=PINNED_BASE_REVISION,
                filename=BASE_SAFETENSORS_INDEX_NAME,
                dest=index_dest,
                token=token,
                within=base_dir,
                allow_download=False,
            )
        )
        shards = parse_index_shards(index_dest, within=base_dir)
    except B3VerifyError as exc:
        return {
            "ok": False,
            "mode": "validate_existing_source_bound",
            "failures": [str(exc)],
            "bindings": bindings,
            "timestamp_utc": utc_now(),
        }

    ordered_base: list[str] = []
    seen: set[str] = set()
    for name in list(REQUIRED_BASE_META_FILES) + shards:
        if name not in seen:
            seen.add(name)
            ordered_base.append(name)

    for name in ordered_base:
        try:
            bindings.append(
                ensure_materialized_file(
                    repo_id=BASE_MODEL_ID,
                    revision=PINNED_BASE_REVISION,
                    filename=name,
                    dest=base_dir / name,
                    token=token,
                    within=base_dir,
                    allow_download=False,
                )
            )
        except B3VerifyError as exc:
            failures.append(str(exc))

    for sf in expected_ten_units():
        unit_dir = adapter_root.joinpath(*sf.split("/"))
        try:
            assert_dir_no_symlink(unit_dir, within=adapter_root)
        except B3VerifyError as exc:
            failures.append(str(exc))
            continue
        for name in REQUIRED_ADAPTER_LOAD_FILES:
            try:
                bindings.append(
                    ensure_materialized_file(
                        repo_id=DENSE_ADAPTER_REPO,
                        revision=PINNED_ADAPTER_REVISION,
                        filename=f"{sf}/{name}",
                        dest=unit_dir / name,
                        token=token,
                        within=adapter_root,
                        allow_download=False,
                    )
                )
            except B3VerifyError as exc:
                failures.append(str(exc))

    return {
        "ok": not failures,
        "mode": "validate_existing_source_bound",
        "failures": failures,
        "bindings": bindings,
        "n_bindings": len(bindings),
        "timestamp_utc": utc_now(),
        "production_pin": PINNED_CHECKPOINT_MANIFEST_SHA256,
        "notes": [
            "no download — existing Volume bytes only",
            "downloader self-check is not independent Agent 1 audit",
        ],
    }


def run_materialize_and_verify(
    *,
    root: Path,
    token: str,
    evidence_dir: Path,
) -> dict[str, Any]:
    """Download/validate required files with source binding; write candidate artifacts."""
    gates = require_billable_gates()
    base_dir = root / "base" / PINNED_BASE_REVISION / "materialized"
    adapter_root = root / "adapters" / PINNED_ADAPTER_REVISION / "materialized"
    base_dir.mkdir(parents=True, exist_ok=True)
    adapter_root.mkdir(parents=True, exist_ok=True)
    assert_dir_no_symlink(base_dir)
    assert_dir_no_symlink(adapter_root)

    source_bindings: list[dict[str, Any]] = []

    index_dest = base_dir / BASE_SAFETENSORS_INDEX_NAME
    source_bindings.append(
        ensure_materialized_file(
            repo_id=BASE_MODEL_ID,
            revision=PINNED_BASE_REVISION,
            filename=BASE_SAFETENSORS_INDEX_NAME,
            dest=index_dest,
            token=token,
            within=base_dir,
        )
    )
    shards = parse_index_shards(index_dest, within=base_dir)
    ordered_base: list[str] = []
    seen: set[str] = set()
    for name in list(REQUIRED_BASE_META_FILES) + shards:
        if name not in seen:
            seen.add(name)
            ordered_base.append(name)

    for name in ordered_base:
        dest = base_dir / name
        source_bindings.append(
            ensure_materialized_file(
                repo_id=BASE_MODEL_ID,
                revision=PINNED_BASE_REVISION,
                filename=name,
                dest=dest,
                token=token,
                within=base_dir,
            )
        )

    for sf in expected_ten_units():
        unit_dir = adapter_root.joinpath(*sf.split("/"))
        if unit_dir.is_symlink():
            raise B3VerifyError(f"symlinked unit directory refused: {sf}")
        unit_dir.mkdir(parents=True, exist_ok=True)
        assert_dir_no_symlink(unit_dir, within=adapter_root)
        for name in REQUIRED_ADAPTER_LOAD_FILES:
            dest = unit_dir / name
            source_bindings.append(
                ensure_materialized_file(
                    repo_id=DENSE_ADAPTER_REPO,
                    revision=PINNED_ADAPTER_REVISION,
                    filename=f"{sf}/{name}",
                    dest=dest,
                    token=token,
                    within=adapter_root,
                )
            )

    report, candidate = inventory_materialized_snapshot(
        base_dir=base_dir, adapter_root=adapter_root
    )
    report.budget = gates
    report.mode = "download_verify_source_bound"
    report.notes.append(
        f"source_bindings={len(source_bindings)} "
        f"(reused={sum(1 for b in source_bindings if b.get('action','').startswith('reused'))})"
    )
    paths = write_candidate_artifacts(report, candidate, evidence_dir)
    bindings_path = evidence_dir / "B3_SOURCE_CONTENT_BINDINGS.json"
    bindings_path.write_text(
        json.dumps(
            {
                "timestamp_utc": utc_now(),
                "bindings": source_bindings,
                "n_bindings": len(source_bindings),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    pointer = {
        "base_materialized": str(base_dir),
        "adapter_materialized": str(adapter_root),
        "base_revision": PINNED_BASE_REVISION,
        "adapter_revision": PINNED_ADAPTER_REVISION,
        "shards": shards,
        "ok": report.ok,
        "candidate_manifest_sha256": report.candidate_manifest_sha256,
        "artifact_paths": {**paths, "source_bindings": str(bindings_path)},
        "budget": gates,
        "timestamp_utc": utc_now(),
        "source_binding_required": True,
    }
    pointer_path = evidence_dir / "B3_CANDIDATE_SNAPSHOT_POINTER.json"
    pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")
    return {
        "ok": report.ok,
        "failures": report.failures,
        "candidate_manifest_sha256": report.candidate_manifest_sha256,
        "artifact_paths": {**paths, "source_bindings": str(bindings_path)},
        "pointer_path": str(pointer_path),
        "n_files_hashed": len(report.files),
        "adapter_weight_distinct": report.adapter_weight_distinct,
        "n_source_bindings": len(source_bindings),
        "budget": gates,
    }
