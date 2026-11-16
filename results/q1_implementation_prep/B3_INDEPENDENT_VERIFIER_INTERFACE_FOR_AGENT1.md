# B3 independent verifier interface (for Agent 1)

**Purpose:** specify the contract Cursor will execute **verbatim** on Modal against the existing Volume.  
**Status:** interface + empty SHA gate prepared; **no Modal verifier job has been run**.

## Constraints

- Mount: `modal.Volume.from_name("traces-q1-b3-checkpoint-bytes").with_mount_options(read_only=True)`
- Call `Volume.reload()` before reads; **no writes** to the source Volume
- CPU only; no GPU; no `from_pretrained`; no checkpoint redownload
- Do **not** import downloader inventory (`inventory_materialized_snapshot`, `run_materialize_and_verify`, `ensure_materialized_file`)
- Prefer stdlib `hashlib` for local digests; `HfApi.get_paths_info` only for source OIDs
- Cumulative B3 spend remains under the owner **$10** cap
- Output: return a JSON-serializable `dict` (runner publishes redacted `B3_INDEPENDENT_VOLUME_READONLY_AUDIT.json` to GitHub)

## Entry point

File slot: `scripts/q1_b3_agent1_independent_volume_verifier.py`

```python
def run_independent_volume_audit(
    *,
    mount_path: Path,          # /mnt/b3
    pointer_path: Path,        # published B3_CANDIDATE_SNAPSHOT_POINTER.json
    hf_token: str | None,      # for get_paths_info only
    volume_name: str,
    verifier_source_sha256: str,
    runner_attestation: dict,
) -> dict:
    ...
```

## Required checks (summary)

1. Validate pointer paths lie under the read-only mount.
2. Lexical no-symlink walk for base + ten unit trees.
3. Rehash all **65** inventory files (chunked SHA-256 + sizes).
4. Parse **Volume** `model.safetensors.index.json` `weight_map` → shard names.
5. Ten pairwise-distinct adapter-weight digests; §11 pins × 10.
6. Source OID bind per file (LFS sha256 or Git blob SHA-1).
7. Independently rebuild candidate manifest via frozen canonicalization; compare to `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` or explain difference (REVISE/BLOCK — never auto-pin).
8. Set `independent_pass_claimed=False` when Cursor is the runner; record limitation text.

## Gate

Runner constant `APPROVED_VERIFIER_SHA256` must equal the published file digest before `audit_main` will launch. Currently **unset**.

## Pointer reference (Volume paths)

From published `B3_CANDIDATE_SNAPSHOT_POINTER.json`:

- base: `/mnt/b3/snapshots/base/c5ebcd40d208330abc697524c919956e692655cf/materialized`
- adapters: `/mnt/b3/snapshots/adapters/6303a74fc832a34d584999364817268aefd9623b/materialized`
- candidate (downloader self-check, not independent): `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55`
