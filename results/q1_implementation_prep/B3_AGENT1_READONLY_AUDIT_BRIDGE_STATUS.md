# B3 Agent 1 — Read-only Modal audit bridge status (Cursor STOP)

**Date:** 2026-11-16  
**Branch:** `q1-impl-audit`  
**Baseline before remediation:** `396758ab005eb19626237aa0c616d79a75fbb4a8`  
**Scientific authorization:** OFF. Production pin remains unset. No merge to `main`.  
**Budget:** cumulative **$10** B3 cap; no GPU; no unnecessary checkpoint redownload.

## Workstream 1 — remediation (Cursor)

Corrected in `src/traces/q1/b3_modal_byte_verify.py` (program version **0.2.0**):

1. **Directory-symlink escape (REVISE defect 1):** lexical component walk + fail-closed resolve containment; refuse symlinked trust root / parents / unit dirs; inventory inspects dirs separately.
2. **Source content-hash binding (REVISE defect 2):** every materialized file must match Hub `get_paths_info` identity at the pinned revision (LFS `lfs.sha256` ↔ local SHA-256, or non-LFS `blob_id` ↔ Git blob OID SHA-1). Bare `dest.exists()` reuse is refused until binding succeeds. Existing Volume bytes can be validated with **no download** via `validate_existing_materialized_source_bound`.
3. Adversarial CPU-only tests added in `tests/test_q1_b3_modal_byte_verify.py`.

**Local CPU test command (no Modal spend):**

```bash
.venv/bin/python -m pytest tests/test_q1_b3_modal_byte_verify.py -q
```

Result: **34 passed**. Production pin still `None`; F1 stream digest unchanged (`fe464187…`).

This is a **downloader-path fix**, not an independent byte PASS.

## Workstream 2 — bridge prepared; verifier **not** run

| Artifact | Role |
|---|---|
| `scripts/q1_b3_modal_readonly_independent_audit.py` | Modal runner: Volume `with_mount_options(read_only=True)`, `reload()`, SHA gate, returns JSON only |
| `scripts/q1_b3_agent1_independent_volume_verifier.py` | **Placeholder slot** — raises until Agent 1 authors/approves body |
| Placeholder source SHA-256 | `e0c722f4c2618d3fb6af7516581748e851569972692d0491769f85fb5a146ef9` (not an approved gate) |
| `APPROVED_VERIFIER_SHA256` in runner | **Empty** — gate closed; Modal audit launch refused |

**STOP:** Cursor has **not** launched any independent verifier Modal job. Agent 1 must author or explicitly approve the verifier, publish its immutable source SHA-256, and open the gate before any delegated run.

## What Agent 1 should do next

1. Review Workstream 1 code + adversarial tests (PASS/REVISE/BLOCK on the two defects).
2. Author (preferred) or review/approve the independent verifier implementing `run_independent_volume_audit(...)` per the interface in the placeholder docstring / handoff.
3. Publish verifier source SHA-256. Cursor will set `APPROVED_VERIFIER_SHA256` to that exact digest and may then run the approved script **unchanged** under the $10 cap.
4. After a delegated run, Agent 1 judges published `B3_INDEPENDENT_VOLUME_READONLY_AUDIT.json` + redacted logs as **delegated remote execution audited** — not “Cursor ran it ⇒ independent PASS.”

## Explicit non-claims

- No independent Volume byte PASS.
- No production pin recommendation.
- No claim that the placeholder verifier authenticates Volume bytes.
- Prior candidate `d070262e…` remains **not owner-frozen** pending independent audit.
