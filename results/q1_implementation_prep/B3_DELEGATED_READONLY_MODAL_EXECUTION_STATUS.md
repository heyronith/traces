# B3 Delegated Read-Only Modal Execution — STOP for Agent 1

**Document type:** delegated execution evidence; **NOT** an owner authorization or production pin.  
**Status:** Agent 1–approved verifier was executed **unchanged** by Cursor as Modal runner against Volume `traces-q1-b3-checkpoint-bytes` (`read_only=True`). **STOP for Agent 1's final independent PASS/REVISE/BLOCK.**  
**Scientific authorization:** OFF. `PINNED_CHECKPOINT_MANIFEST_SHA256` remains unset. No model forwards/GPU/checkpoint redownload/merge to `main`.

## Provenance

| Item | Value |
|---|---|
| Baseline at launch | `7536fdaf2780dc7902b884b61da4f5901a54e80d` |
| Approved verifier SHA-256 | `83eb9656700adc82be96beff289612f5aba05e43fe23a3c5c0aac6b4ffb50afa` |
| Local SHA at run | matched gate exactly |
| Modal app | `traces-q1-b3-readonly-independent-audit` |
| Modal run ID | `ap-kFkQ64sF5s5mmPCTVRChYl` |
| Run URL | https://modal.com/apps/heyronith/main/ap-kFkQ64sF5s5mmPCTVRChYl |
| Window | `2026-11-17T00:06:22Z` → `2026-11-17T00:07:27Z` (~65s) |

## Verifier-level outcome (not Agent 1 final verdict)

From `B3_INDEPENDENT_VOLUME_READONLY_AUDIT.json`:

- `ok=true`, `verdict=PASS` (**verifier-level only**; `independent_pass_claimed=false`)
- 65 files hashed; all source OID matches; source binding complete
- Candidate rebuilt digest `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` matches expected
- Adapter weights pairwise distinct; §11 `all_match=true`
- Symlink checks: no symlinks/specials; **`extras=[]`** for base and adapters
- **`warnings.extra_hidden`:** not present / no hidden-extra inventory warnings recorded
- Sole warning: `statvfs does not report ST_RDONLY (runner attests read_only)` — runner used `with_mount_options(read_only=True)`; Agent 1 should weigh this limitation
- `report_body_sha256=651e39366c093646deb6527a7c95a98978f49175910a3c51f6bef93cc61052e6` (recomputed locally after stripping `report_body_sha256` + runner `volume_read_only`)

## Cost (within cumulative $10 B3 cap)

| Meter | USD |
|---|---|
| This job Modal metered (CPU+Memory+Egress) | **~$0.001687** |
| Preflight timeout safety bound | $0.2379 |
| Workspace billed cost after job (`modal billing summary`) | **$0.00** |

Details: `B3_MODAL_BILLING_TODAY_SNAPSHOT.json`, `B3_DELEGATED_READONLY_MODAL_EXECUTION_META.json`.

## Published artifacts

- `B3_INDEPENDENT_VOLUME_READONLY_AUDIT.json` (full verifier return)
- `B3_INDEPENDENT_VOLUME_READONLY_AUDIT_MODAL_RUN.log` (redacted)
- `B3_MODAL_BILLING_TODAY_SNAPSHOT.json`
- `B3_DELEGATED_READONLY_MODAL_EXECUTION_META.json`
- this status note

## Explicit non-claims

- Cursor delegated execution ≠ Agent 1 independent PASS.
- No production pin recommendation.
- No scientific-execution authorization.
