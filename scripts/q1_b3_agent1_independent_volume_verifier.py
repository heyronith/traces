#!/usr/bin/env python3
"""Agent 1 independent, read-only B3 Volume verifier (TRACES Q1).

Authored by Agent 1 (independent auditor). Executed verbatim by a delegated
runner under a SHA-256 gate. Delegated execution is NOT direct Agent 1 Modal
access; this module therefore never claims an independent PASS
(``independent_pass_claimed`` is always False). Agent 1 issues the verdict after
inspecting the published report.

Independence rules honoured here:
  * Standard library only, plus ``huggingface_hub.HfApi.get_paths_info`` for
    source object identities (imported lazily). No ``traces.*`` imports; no
    downloader inventory, hashing, or path helpers.
  * Every frozen constant is restated here from the frozen protocol / pins,
    not imported from the code under audit.
  * Paths are derived from ``mount_path`` + pinned revisions; the published
    pointer is only cross-checked, never trusted for locations.
  * Read-only: files are opened O_RDONLY|O_NOFOLLOW; nothing is written,
    downloaded, loaded into a model, or executed. CPU only.

Checks:
  1. Pointer cross-check (paths/revisions) and mount-to-trust-root lstat chain.
  2. Full no-follow tree walk of the base and adapter trust roots: no symlinks,
     no special files, no forbidden tokenizer files, exact inventory.
  3. Streaming SHA-256 (+ Git blob SHA-1 for small files) of all 65 files with
     before/after stat stability.
  4. Parse the Volume's own model.safetensors.index.json bytes; exact shard
     inventory; each shard's safetensors header tensor set equals its weight_map
     share; header offsets cover the data buffer exactly.
  5. Ten adapters: six required files each, distinct weight digests, valid
     safetensors headers, family-consistent LoRA layer sets (l19 -> {19};
     l1523 -> within 15..23 and including 15 and 23).
  6. Frozen protocol section 11 tokenizer/template pins for all ten units.
  7. Hub source binding at immutable pinned commits: LFS sha256 vs local
     SHA-256, or Git blob SHA-1 vs local blob OID; sizes must match.
  8. Independent rebuild of the candidate manifest; canonical SHA-256 compared
     with the frozen candidate digest; diff against the Volume's evidence copy.

Verdicts: PASS (verifier-level, no failures), BLOCK (any integrity failure),
INCOMPLETE (only operational failures, e.g. Hub unreachable). Never REVISE
automatically; never pins anything.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import platform
import re
import stat
import struct
import sys
import traceback
from pathlib import Path
from typing import Any

VERIFIER_NAME = "agent1_independent_b3_volume_verifier"
VERIFIER_VERSION = "1.0.0"

# --- Frozen identities (restated independently; do not import) -------------
BASE_REPO = "google/gemma-2-2b"
BASE_REVISION = "c5ebcd40d208330abc697524c919956e692655cf"
ADAPTER_REPO = "interpretable-finetuning/topklora-gemma-2-2b-dense-lora"
ADAPTER_REVISION = "6303a74fc832a34d584999364817268aefd9623b"

UNITS = tuple(
    [f"l19/seed{s}" for s in (42, 43, 44, 45, 46)]
    + [f"l1523/seed{s}" for s in (42, 43, 44, 45, 46)]
)
ADAPTER_REQUIRED = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)
ADAPTER_WEIGHT = "adapter_model.safetensors"
FORBIDDEN_BASENAMES = frozenset({"tokenizer.json", "tokenizer.model.json"})
BASE_META = ("config.json", "model.safetensors.index.json")
BASE_INDEX = "model.safetensors.index.json"
EXPECTED_SHARDS = (
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
)
EXPECTED_FILE_COUNT = 65

# Frozen protocol v1.3 section 11 tokenizer/template pins.
SECTION11_PINS = {
    "tokenizer.model": "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2",
    "tokenizer_config.json": "5db39e7ccbeceb1dc1136ff54d717f6f84f3556893e6cc30ee4972c408ed4dc2",
    "chat_template.jinja": "ecd6ae513fe103f0eb62e8ab5bfa8d0fe45c1074fa398b089c93a7e70c15cfd6",
    "special_tokens_map.json": "5b36db1ae0240191639152e5edfc68785230efb3b3cbb3ba546687a5a9e9c680",
}

# Candidate manifest under audit (downloader self-check; not trusted).
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55"
)
MANIFEST_ARTIFACT_TYPE = "q1_checkpoint_provenance_manifest_v1"
MANIFEST_NOTES = [
    "B3_CANDIDATE_ONLY — not production authorization",
    "PINNED_CHECKPOINT_MANIFEST_SHA256 must remain unset until owner freeze",
]
MANIFEST_EXCLUDED_KEYS = frozenset({"signature", "detached_signature"})

CHUNK = 4 * 1024 * 1024
GIT_SHA1_MAX_BYTES = 64 * 1024 * 1024  # all non-LFS files are far smaller
SAFETENSORS_MAX_HEADER = 100 * 1024 * 1024
HEX64 = re.compile(r"^[0-9a-f]{64}$")
HEX40 = re.compile(r"^[0-9a-f]{40}$")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
SHARD_RE = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")


class _Report:
    def __init__(self) -> None:
        self.failures: list[dict[str, str]] = []  # integrity
        self.operational: list[dict[str, str]] = []  # cannot evaluate
        self.warnings: list[dict[str, str]] = []

    def fail(self, check: str, msg: str) -> None:
        self.failures.append({"check": check, "detail": msg})

    def op(self, check: str, msg: str) -> None:
        self.operational.append({"check": check, "detail": msg})

    def warn(self, check: str, msg: str) -> None:
        self.warnings.append({"check": check, "detail": msg})


def _utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _redact(text: str, token: str | None) -> str:
    text = re.sub(r"hf_[A-Za-z0-9]{6,}", "[REDACTED_HF_TOKEN]", text)
    if token:
        text = text.replace(token, "[REDACTED_HF_TOKEN]")
    return text


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


# --- Filesystem primitives (no-follow) ---------------------------------------


def _lstat_kind(path: Path) -> str:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return "missing"
    m = st.st_mode
    if stat.S_ISLNK(m):
        return "symlink"
    if stat.S_ISDIR(m):
        return "dir"
    if stat.S_ISREG(m):
        return "file"
    return "special"


def _check_chain(mount: Path, target: Path, rep: _Report, label: str) -> bool:
    """lstat every component strictly below the mount down to target."""
    try:
        rel = target.relative_to(mount)
    except ValueError:
        rep.fail("path_containment", f"{label}: {target} not under mount {mount}")
        return False
    if ".." in rel.parts:
        rep.fail("path_containment", f"{label}: '..' component in {target}")
        return False
    cur = mount
    ok = True
    for part in rel.parts:
        cur = cur / part
        kind = _lstat_kind(cur)
        if kind != "dir":
            rep.fail("symlink_chain", f"{label}: component {cur} is {kind}, need real dir")
            ok = False
            break
    if ok:
        try:
            if not str(target.resolve(strict=True)).startswith(
                str(mount.resolve(strict=True)).rstrip("/") + "/"
            ):
                rep.fail("path_containment", f"{label}: resolved {target} escapes mount")
                ok = False
        except OSError as exc:
            rep.fail("path_containment", f"{label}: resolve failed: {exc}")
            ok = False
    return ok


def _walk(root: Path) -> list[tuple[str, str]]:
    """No-follow recursive walk -> [(relpath, kind)]. Never descends symlinks."""
    out: list[tuple[str, str]] = []
    stack = [root]
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                p = Path(e.path)
                rel = p.relative_to(root).as_posix()
                if e.is_symlink():
                    out.append((rel, "symlink"))
                elif e.is_dir(follow_symlinks=False):
                    out.append((rel, "dir"))
                    stack.append(p)
                elif e.is_file(follow_symlinks=False):
                    out.append((rel, "file"))
                else:
                    out.append((rel, "special"))
    return sorted(out)


def _hash_file(path: Path) -> dict[str, Any]:
    """Streaming SHA-256 (+ Git blob SHA-1 if small) with no-follow open and
    before/after stability check."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        st0 = os.fstat(fd)
        if not stat.S_ISREG(st0.st_mode):
            raise OSError(f"not a regular file: {path}")
        size = st0.st_size
        h256 = hashlib.sha256()
        h1 = None
        if size <= GIT_SHA1_MAX_BYTES:
            h1 = hashlib.sha1()
            h1.update(b"blob " + str(size).encode("ascii") + b"\0")
        total = 0
        with os.fdopen(os.dup(fd), "rb", buffering=0) as fh:
            while True:
                chunk = fh.read(CHUNK)
                if not chunk:
                    break
                h256.update(chunk)
                if h1 is not None:
                    h1.update(chunk)
                total += len(chunk)
        st1 = os.fstat(fd)
    finally:
        os.close(fd)
    st2 = os.lstat(path)
    stable = (
        total == size
        and (st0.st_size, st0.st_mtime_ns, st0.st_ino)
        == (st1.st_size, st1.st_mtime_ns, st1.st_ino)
        == (st2.st_size, st2.st_mtime_ns, st2.st_ino)
    )
    return {
        "sha256": h256.hexdigest(),
        "git_blob_sha1": h1.hexdigest() if h1 is not None else None,
        "size_bytes": total,
        "stat_size": size,
        "stable": stable,
    }


def _read_small(path: Path, limit: int = 8 * 1024 * 1024) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        with os.fdopen(os.dup(fd), "rb") as fh:
            data = fh.read(limit + 1)
    finally:
        os.close(fd)
    if len(data) > limit:
        raise OSError(f"file larger than {limit} bytes: {path}")
    return data


def _safetensors_header(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        size = os.fstat(fd).st_size
        with os.fdopen(os.dup(fd), "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError("file shorter than 8 bytes")
            (n,) = struct.unpack("<Q", raw)
            if n < 2 or n > SAFETENSORS_MAX_HEADER or 8 + n > size:
                raise ValueError(f"implausible header length {n} for size {size}")
            hdr = fh.read(n)
    finally:
        os.close(fd)
    meta = json.loads(hdr.decode("utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("header is not a JSON object")
    buf = size - 8 - n
    spans = []
    tensors: dict[str, Any] = {}
    for name, info in meta.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or "data_offsets" not in info:
            raise ValueError(f"tensor {name!r} lacks data_offsets")
        a, b = info["data_offsets"]
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b <= buf):
            raise ValueError(f"tensor {name!r} offsets out of range")
        spans.append((a, b))
        tensors[name] = {"dtype": info.get("dtype"), "shape": info.get("shape")}
    spans.sort()
    pos = 0
    for a, b in spans:
        if a != pos:
            raise ValueError(f"buffer gap/overlap at {pos} (next span starts {a})")
        pos = b
    if pos != buf:
        raise ValueError(f"tensor spans cover {pos} of {buf} buffer bytes")
    return {"tensors": tensors, "header_len": n, "buffer_len": buf}


# --- Hub source identities ----------------------------------------------------


def _source_identities(
    repo: str, revision: str, paths: list[str], token: str | None
) -> dict[str, dict[str, Any]]:
    from huggingface_hub import HfApi  # lazy; only for source identities

    api = HfApi(token=token)
    out: dict[str, dict[str, Any]] = {}
    for i in range(0, len(paths), 50):
        batch = paths[i : i + 50]
        infos = api.get_paths_info(repo, batch, revision=revision, repo_type="model")
        for info in infos:
            p = getattr(info, "path", None)
            if p is None:
                continue
            lfs = getattr(info, "lfs", None)
            lfs_sha = None
            if lfs is not None:
                lfs_sha = getattr(lfs, "sha256", None)
                if lfs_sha is None and isinstance(lfs, dict):
                    lfs_sha = lfs.get("sha256")
            out[str(p)] = {
                "is_dir": getattr(info, "tree_id", None) is not None
                and getattr(info, "blob_id", None) is None,
                "size": getattr(info, "size", None),
                "blob_id": getattr(info, "blob_id", None),
                "lfs_sha256": lfs_sha,
            }
    return out


def _bind(local: dict[str, Any], src: dict[str, Any] | None) -> tuple[bool, str, dict]:
    if src is None:
        return False, "no source entry returned by Hub", {}
    if src.get("is_dir"):
        return False, "source path is a directory", {}
    rec = {"source_size": src.get("size")}
    if src.get("size") is not None and int(src["size"]) != local["size_bytes"]:
        return False, f"size local={local['size_bytes']} source={src['size']}", rec
    if src.get("lfs_sha256") is not None:
        s = str(src["lfs_sha256"]).lower()
        rec.update(kind="lfs_sha256", source_digest=s)
        if not HEX64.match(s):
            return False, "malformed source LFS sha256", rec
        return (s == local["sha256"]), ("ok" if s == local["sha256"] else "LFS sha256 mismatch"), rec
    b = str(src.get("blob_id") or "").lower()
    rec.update(kind="git_blob_sha1", source_digest=b)
    if not HEX40.match(b):
        return False, "missing/malformed source blob_id", rec
    if local.get("git_blob_sha1") is None:
        return False, "non-LFS source but local blob SHA-1 not computed", rec
    ok = b == local["git_blob_sha1"]
    return ok, ("ok" if ok else "Git blob SHA-1 mismatch"), rec


# --- Main audit ---------------------------------------------------------------


def _audit(
    *,
    mount_path: Path,
    pointer_path: Path,
    hf_token: str | None,
    volume_name: str,
    verifier_source_sha256: str,
    runner_attestation: dict[str, Any],
) -> dict[str, Any]:
    rep = _Report()
    t0 = _utc()
    mount = Path(mount_path)
    base_root = mount / "snapshots" / "base" / BASE_REVISION / "materialized"
    adapter_root = mount / "snapshots" / "adapters" / ADAPTER_REVISION / "materialized"
    result: dict[str, Any] = {
        "verifier": VERIFIER_NAME,
        "verifier_version": VERIFIER_VERSION,
        "verifier_source_sha256": verifier_source_sha256,
        "volume_name": volume_name,
        "started_utc": t0,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "runner_attestation": dict(runner_attestation or {}),
        "independent_pass_claimed": False,
        "execution_mode": "delegated_remote_execution",
        "limitations": [
            "Executed by a delegated runner, not by Agent 1 with direct Modal access.",
            "Runner integrity (gate check, read-only mount, unmodified report) is "
            "attested by the runner, not independently observed by Agent 1.",
            "This report is evidence for Agent 1's verdict; it is not itself a PASS.",
        ],
        "paths": {"base_root": str(base_root), "adapter_root": str(adapter_root)},
    }

    # 0. Mount sanity / read-only attestation
    if _lstat_kind(mount) not in ("dir", "symlink") or not mount.is_dir():
        rep.op("mount", f"mount path {mount} is not a directory")
    try:
        ro = bool(os.statvfs(mount).f_flag & getattr(os, "ST_RDONLY", 1))
        result["mount_statvfs_read_only"] = ro
        if not ro:
            rep.warn("mount", "statvfs does not report ST_RDONLY (runner attests read_only)")
    except OSError as exc:
        rep.warn("mount", f"statvfs failed: {exc}")

    # 1. Pointer cross-check (never used for locations)
    try:
        ptr = json.loads(_read_small(Path(pointer_path)).decode("utf-8"))
        exp = {
            "base_materialized": str(base_root),
            "adapter_materialized": str(adapter_root),
            "base_revision": BASE_REVISION,
            "adapter_revision": ADAPTER_REVISION,
            "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        }
        for k, v in exp.items():
            if ptr.get(k) != v:
                rep.fail("pointer", f"pointer {k}={ptr.get(k)!r} expected {v!r}")
        if list(ptr.get("shards") or []) != list(EXPECTED_SHARDS):
            rep.fail("pointer", f"pointer shards {ptr.get('shards')!r} unexpected")
        result["pointer_evidence_paths"] = ptr.get("artifact_paths")
    except Exception as exc:  # noqa: BLE001
        ptr = {}
        rep.op("pointer", f"cannot read pointer: {exc}")

    # 2. Chains + full no-follow walks
    sym: dict[str, Any] = {}
    roots_ok = True
    for label, root in (("base", base_root), ("adapters", adapter_root)):
        if not _check_chain(mount, root, rep, label):
            roots_ok = False
            continue
        entries = _walk(root)
        sym[label] = {
            "n_entries": len(entries),
            "symlinks": [r for r, k in entries if k == "symlink"],
            "special": [r for r, k in entries if k == "special"],
        }
        for r, k in entries:
            name = r.rsplit("/", 1)[-1]
            if k == "symlink":
                rep.fail("symlink", f"{label}: symlink {r}")
            elif k == "special":
                rep.fail("special_file", f"{label}: special file {r}")
            if name in FORBIDDEN_BASENAMES:
                rep.fail("forbidden_file", f"{label}: forbidden {r}")
        files = {r for r, k in entries if k == "file"}
        dirs = {r for r, k in entries if k == "dir"}
        if label == "base":
            expected_files = set(BASE_META) | set(EXPECTED_SHARDS)
            expected_dirs: set[str] = set()
        else:
            expected_files = {f"{u}/{n}" for u in UNITS for n in ADAPTER_REQUIRED}
            expected_dirs = {"l19", "l1523"} | set(UNITS)
        for r in sorted(files - expected_files):
            top = r.split("/")
            if any(p.startswith(".") for p in top):
                rep.warn("extra_hidden", f"{label}: hidden extra file {r}")
            else:
                rep.fail("extra_file", f"{label}: non-inventory file {r}")
        for r in sorted(dirs - expected_dirs):
            if any(p.startswith(".") for p in r.split("/")):
                rep.warn("extra_hidden", f"{label}: hidden extra dir {r}")
            else:
                rep.fail("extra_dir", f"{label}: unexpected directory {r}")
        for r in sorted(expected_files - files):
            rep.fail("missing_file", f"{label}: missing {r}")
        sym[label]["extras"] = sorted((files - expected_files) | (dirs - expected_dirs))
    result["symlink_checks"] = sym
    if not roots_ok:
        return _finish(result, rep, hf_token)

    # 3. Hash all inventory files
    inventory: list[tuple[str, str, Path, str, str]] = []  # (role, key_rel, path, repo, src_path)
    for n in BASE_META + EXPECTED_SHARDS:
        inventory.append(("base", n, base_root / n, BASE_REPO, n))
    for u in UNITS:
        for n in ADAPTER_REQUIRED:
            rel = f"{u}/{n}"
            inventory.append(("adapter", rel, adapter_root / u / n, ADAPTER_REPO, rel))
    hashes: dict[str, dict[str, Any]] = {}
    for role, rel, p, _repo, _sp in inventory:
        if _lstat_kind(p) != "file":
            rep.fail("hash", f"{role}:{rel} not a regular file")
            continue
        try:
            h = _hash_file(p)
        except Exception as exc:  # noqa: BLE001
            rep.fail("hash", f"{role}:{rel}: {exc}")
            continue
        if not h["stable"]:
            rep.fail("hash_stability", f"{role}:{rel} changed or short read during hashing")
        hashes[f"{role}:{rel}"] = h
    result["n_files_hashed"] = len(hashes)
    if len(inventory) != EXPECTED_FILE_COUNT or len(hashes) != EXPECTED_FILE_COUNT:
        rep.fail(
            "inventory_count",
            f"expected {EXPECTED_FILE_COUNT} files, inventory={len(inventory)} hashed={len(hashes)}",
        )

    # 4. Index parse from Volume bytes + shard headers
    idx: dict[str, Any] = {}
    try:
        raw = _read_small(base_root / BASE_INDEX)
        if hashlib.sha256(raw).hexdigest() != hashes.get(f"base:{BASE_INDEX}", {}).get("sha256"):
            rep.fail("index", "index bytes changed between hash and parse")
        data = json.loads(raw.decode("utf-8"))
        wm = data.get("weight_map") if isinstance(data, dict) else None
        if not isinstance(wm, dict) or not wm:
            raise ValueError("weight_map missing or empty")
        shard_of: dict[str, list[str]] = {}
        for tname, shard in wm.items():
            if not isinstance(shard, str) or "/" in shard or "\\" in shard or shard in ("", ".", ".."):
                raise ValueError(f"unsafe shard name {shard!r} for {tname!r}")
            shard_of.setdefault(shard, []).append(tname)
        shards = sorted(shard_of)
        idx = {
            "n_tensors": len(wm),
            "shards": shards,
            "tensors_per_shard": {s: len(v) for s, v in shard_of.items()},
            "metadata": data.get("metadata"),
        }
        if tuple(shards) != EXPECTED_SHARDS:
            rep.fail("index", f"index shard set {shards} != frozen {list(EXPECTED_SHARDS)}")
        for s in shards:
            m = SHARD_RE.match(s)
            if not m or int(m.group(2)) != len(shards):
                rep.fail("index", f"shard name {s!r} inconsistent with {len(shards)} shards")
        hdr_res = {}
        for s in EXPECTED_SHARDS:
            try:
                hdr = _safetensors_header(base_root / s)
                names = set(hdr["tensors"])
                want = set(shard_of.get(s, []))
                hdr_res[s] = {
                    "n_tensors": len(names),
                    "matches_weight_map": names == want,
                    "buffer_len": hdr["buffer_len"],
                }
                if names != want:
                    rep.fail(
                        "shard_header",
                        f"{s}: header tensors differ from weight_map "
                        f"(only_header={len(names - want)}, only_index={len(want - names)})",
                    )
            except Exception as exc:  # noqa: BLE001
                rep.fail("shard_header", f"{s}: {exc}")
        idx["shard_headers"] = hdr_res
    except Exception as exc:  # noqa: BLE001
        rep.fail("index", f"index parse failed: {exc}")
    result["index"] = idx

    # 5/6. Adapters + section 11
    adapters: dict[str, Any] = {}
    sec11: dict[str, Any] = {}
    weight_digests: dict[str, str] = {}
    for u in UNITS:
        fam = u.split("/")[0]
        info: dict[str, Any] = {"family": fam}
        wkey = f"adapter:{u}/{ADAPTER_WEIGHT}"
        if wkey in hashes:
            weight_digests[u] = hashes[wkey]["sha256"]
        try:
            hdr = _safetensors_header(adapter_root / u / ADAPTER_WEIGHT)
            layers = sorted(
                {int(m.group(1)) for t in hdr["tensors"] for m in [LAYER_RE.search(t)] if m}
            )
            info["n_tensors"] = len(hdr["tensors"])
            info["lora_layers"] = layers
            info["dtypes"] = sorted({str(v["dtype"]) for v in hdr["tensors"].values()})
            if fam == "l19":
                fam_ok = layers == [19]
            else:
                fam_ok = (
                    bool(layers)
                    and set(layers) <= set(range(15, 24))
                    and 15 in layers
                    and 23 in layers
                )
            info["family_layers_ok"] = fam_ok
            if not fam_ok:
                rep.fail("adapter_family", f"{u}: LoRA layers {layers} inconsistent with {fam}")
        except Exception as exc:  # noqa: BLE001
            rep.fail("adapter_header", f"{u}: {exc}")
        try:
            cfg = json.loads(_read_small(adapter_root / u / "adapter_config.json").decode("utf-8"))
            info["adapter_config"] = {
                k: cfg.get(k)
                for k in (
                    "peft_type",
                    "r",
                    "lora_alpha",
                    "target_modules",
                    "layers_to_transform",
                    "base_model_name_or_path",
                    "revision",
                )
            }
        except Exception as exc:  # noqa: BLE001
            rep.fail("adapter_config", f"{u}: unparseable adapter_config.json: {exc}")
        adapters[u] = info
        per = {}
        for n, pin in SECTION11_PINS.items():
            got = hashes.get(f"adapter:{u}/{n}", {}).get("sha256")
            per[n] = got == pin
            if got != pin:
                rep.fail("section11", f"{u}/{n}: {got} != frozen {pin}")
        sec11[u] = per
    if len(weight_digests) != 10 or len(set(weight_digests.values())) != 10:
        rep.fail(
            "adapter_distinct",
            f"adapter weights: {len(weight_digests)} present, "
            f"{len(set(weight_digests.values()))} distinct (need 10/10)",
        )
    result["adapters"] = adapters
    result["section11"] = {"all_match": all(all(v.values()) for v in sec11.values()), "per_unit": sec11}
    result["adapter_weights_pairwise_distinct"] = len(set(weight_digests.values())) == 10

    # 7. Hub source binding
    src_ok = None
    try:
        src = {}
        src.update(
            {("base", k): v for k, v in _source_identities(
                BASE_REPO, BASE_REVISION, list(BASE_META + EXPECTED_SHARDS), hf_token
            ).items()}
        )
        src.update(
            {("adapter", k): v for k, v in _source_identities(
                ADAPTER_REPO,
                ADAPTER_REVISION,
                [f"{u}/{n}" for u in UNITS for n in ADAPTER_REQUIRED],
                hf_token,
            ).items()}
        )
        src_ok = True
    except Exception as exc:  # noqa: BLE001
        src = {}
        rep.op("source_binding", f"Hub get_paths_info failed: {type(exc).__name__}: {exc}")
    files_out = []
    for role, rel, _p, repo, sp in inventory:
        key = f"{role}:{rel}"
        h = hashes.get(key)
        row: dict[str, Any] = {"key": key, "repo": repo, "source_path": sp}
        if h:
            row.update(sha256=h["sha256"], size_bytes=h["size_bytes"], git_blob_sha1=h["git_blob_sha1"])
        if src_ok and h:
            ok, why, rec = _bind(h, src.get((role, sp)))
            row.update(source_match=ok, source_detail=why, **rec)
            if not ok:
                rep.fail("source_binding", f"{key}: {why}")
        else:
            row["source_match"] = None
        files_out.append(row)
    result["files"] = files_out
    result["source_binding_complete"] = bool(src_ok) and all(
        r.get("source_match") is True for r in files_out
    )

    # 8. Rebuild candidate manifest
    try:
        base_files = {f"base:{n}": hashes[f"base:{n}"]["sha256"] for n in BASE_META + EXPECTED_SHARDS}
        units = {}
        for u in UNITS:
            fam, seed_tok = u.split("/")
            units[u] = {
                "family": fam,
                "files": {
                    f"adapter:{u}/{n}": hashes[f"adapter:{u}/{n}"]["sha256"]
                    for n in ADAPTER_REQUIRED
                },
                "seed": int(seed_tok[4:]),
                "subfolder": u,
            }
        manifest = {
            "artifact_type": MANIFEST_ARTIFACT_TYPE,
            "base_files": base_files,
            "notes": list(MANIFEST_NOTES),
            "self_generated": False,
            "units": units,
        }
        digest = hashlib.sha256(_canonical(manifest)).hexdigest()
        result["candidate_manifest_sha256"] = digest
        result["candidate_manifest_matches_expected"] = (
            digest == EXPECTED_CANDIDATE_MANIFEST_SHA256
        )
        result["rebuilt_manifest"] = manifest
        if digest != EXPECTED_CANDIDATE_MANIFEST_SHA256:
            rep.fail(
                "manifest",
                f"rebuilt canonical SHA-256 {digest} != candidate "
                f"{EXPECTED_CANDIDATE_MANIFEST_SHA256}",
            )
        # Diff against the Volume evidence copy (comparison only; not trusted)
        ev = (ptr.get("artifact_paths") or {}).get("candidate_manifest")
        if ev:
            evp = Path(ev)
            if _check_chain(mount, evp.parent, rep, "evidence") and _lstat_kind(evp) == "file":
                vol_man = json.loads(_read_small(evp).decode("utf-8"))
                body = {k: v for k, v in vol_man.items() if k not in MANIFEST_EXCLUDED_KEYS}
                vdig = hashlib.sha256(_canonical(body)).hexdigest()
                diffs = []
                for k, v in base_files.items():
                    if (vol_man.get("base_files") or {}).get(k) != v:
                        diffs.append(k)
                for u, ud in units.items():
                    vf = ((vol_man.get("units") or {}).get(u) or {}).get("files") or {}
                    diffs += [k for k, v in ud["files"].items() if vf.get(k) != v]
                result["volume_evidence_manifest"] = {
                    "canonical_sha256": vdig,
                    "matches_rebuilt": vdig == digest,
                    "differing_keys": diffs,
                }
                if vdig != digest:
                    rep.fail("manifest", f"Volume evidence manifest canonical {vdig} != rebuilt {digest}")
            else:
                rep.warn("manifest", f"Volume evidence manifest not readable as regular file: {ev}")
    except Exception as exc:  # noqa: BLE001
        rep.fail("manifest", f"manifest rebuild failed: {exc}")

    return _finish(result, rep, hf_token)


def _finish(result: dict[str, Any], rep: _Report, token: str | None) -> dict[str, Any]:
    if rep.failures:
        verdict = "BLOCK"
    elif rep.operational:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS"
    result.update(
        verdict=verdict,
        ok=verdict == "PASS",
        verdict_scope="verifier-level only; Agent 1 issues the audit verdict",
        failures=rep.failures,
        operational_failures=rep.operational,
        warnings=rep.warnings,
        independent_pass_claimed=False,
        finished_utc=_utc(),
    )
    text = _redact(json.dumps(result, sort_keys=True, default=str), token)
    clean = json.loads(text)
    clean["report_body_sha256"] = hashlib.sha256(_canonical(clean)).hexdigest()
    return clean


def run_independent_volume_audit(
    *,
    mount_path: Path,
    pointer_path: Path,
    hf_token: str | None,
    volume_name: str,
    verifier_source_sha256: str,
    runner_attestation: dict[str, Any],
) -> dict[str, Any]:
    """Interface entry point. Never raises; never claims an independent PASS."""
    try:
        return _audit(
            mount_path=Path(mount_path),
            pointer_path=Path(pointer_path),
            hf_token=hf_token,
            volume_name=volume_name,
            verifier_source_sha256=verifier_source_sha256,
            runner_attestation=runner_attestation,
        )
    except Exception as exc:  # noqa: BLE001
        rep = _Report()
        rep.op("unhandled", f"{type(exc).__name__}: {exc}")
        return _finish(
            {
                "verifier": VERIFIER_NAME,
                "verifier_version": VERIFIER_VERSION,
                "verifier_source_sha256": verifier_source_sha256,
                "traceback": traceback.format_exc()[-4000:],
                "runner_attestation": dict(runner_attestation or {}),
            },
            rep,
            hf_token,
        )


if __name__ == "__main__":
    raise SystemExit(
        "Library module: invoke run_independent_volume_audit() via the gated runner."
    )
