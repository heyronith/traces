# TRACES Q1 — Agent 1 Approved Verifier: Gated, Delegated Read-Only Modal Execution Handoff

**Document type:** technical execution handoff and engineering provenance; **NOT** an authorization request. Owner approvals are handled in chat, not GitHub.  
**Status:** Agent 1 supplied/approved independent verifier source. **Verifier installed and SHA gate configured; the read-only Modal audit has NOT been run. No B3 independent byte PASS.**  
**Scientific authorization:** **OFF**. Production checkpoint pin remains `None`. No model loads/forwards, GPU, checkpoint redownload, science experiment, protocol edit, or frozen-main merge.

## Immutable provenance

| Anchor | Value |
|---|---|
| Frozen main | `b1e8874f45311678e6fa241f504e7e09d4a7808c` |
| Last remediation/bridge baseline | `f72f9f68bd51a9afb3e49f0e195a7d1c8341586c`; tree `b0edfa7c932e63720a29d56bd527f948103dae3d` |
| Agent 1-approved source filename | `scripts/q1_b3_agent1_independent_volume_verifier.py` |
| **SHA-256 of Agent 1 uploaded, approved source** | **`83eb9656700adc82be96beff289612f5aba05e43fe23a3c5c0aac6b4ffb50afa`** |
| Verified upload | 805 lines; 32,244 UTF-8 bytes; Python syntax compilation passed in ChatGPT container; no `traces.*` imports |
| Verifier source-only publication commit | `c28024d632a6865002663fe7d152cd2302b49698` |
| Gate-open runner publication commit | `9ed5aaf8e6276eb0a218bb97e579d909e2cf3238` |
| Existing candidate canonical SHA-256 (NOT owner frozen) | `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` |
| Volume | `traces-q1-b3-checkpoint-bytes` |
| Approved budget | **$10 USD cumulative B3 cap** in owner chat; downloader reported billed $0.00, not independently metered by Agent 1 |

### Source installation evidence

The user-uploaded approved verifier file was hashed **directly** by ChatGPT's container: `83eb9656700adc82be96beff289612f5aba05e43fe23a3c5c0aac6b4ffb50afa`. It was UTF-8-read and uploaded through the GitHub connector with its original terminal LF preserved; the published GitHub file was fetched back and compared character-for-character to the upload. The only edit to the prepared runner was assigning `APPROVED_VERIFIER_SHA256` to that exact digest; the read-only Volume mount and independent-pass-disabled semantics were preserved.

The previous `B3_AGENT1_READONLY_AUDIT_BRIDGE_STATUS.md` records the *earlier* blocked placeholder gate. This document supersedes the earlier gate status **only**; the independent Volume byte audit still has not occurred.

## Next action for Cursor (not yet run)

1. Fetch branch tip containing this handoff, inspect both scripts and verify the approved verifier's **local SHA-256** exactly `83eb9656700adc82be96beff289612f5aba05e43fe23a3c5c0aac6b4ffb50afa`. **Do not change any byte of the approved verifier.** Refuse to execute if mismatch; do not update gate to match changes without Agent 1 approval.
2. Confirm original `APPROVED_VERIFIER_SHA256` gate in `scripts/q1_b3_modal_readonly_independent_audit.py` matches the source. Run `--mode status` locally and inspect `gate_open=true` before Modal job.
3. Check the **cumulative** $10 owner cap against prior B3 usage/actual account billing. Do not assume that reported $0.00 account charge is proof of no resource usage. Ensure 2 CPU / 8 GiB / 2700s max, `max_containers=1`, no GPU, no downloads and fail-closed costs. A short CPU read-only audit is permitted within the previously approved B3 cap; **STOP** if billed cumulative cost cannot be kept within limit.
4. Use existing Modal workspace, Volume mounted `with_mount_options(read_only=True)`, `reload()`, existing `hf-token` Secret for read-only Hub source metadata lookups only. Do not disclose token values, checkpoint bytes, signed URLs or secret metadata in GitHub. Do not grant Agent 1 Modal access.
5. Run the exact runner entrypoint after gate success:
   ```bash
   B3_OWNER_APPROVED_MAX_USD=10 B3_CHAT_CAP_CONFIRMED=1 \
     modal run scripts/q1_b3_modal_readonly_independent_audit.py::audit_main
   ```
6. Publish the **full** returned `B3_INDEPENDENT_VOLUME_READONLY_AUDIT.json` and non-secret/appropriately redacted Modal logs, interpreter/version, run ID, metered/billed cost, source/gate SHAs and executed code tree to `q1-impl-audit`. The verifier already creates `report_body_sha256`; the runner appends `volume_read_only`. Agent 1's verification procedure: remove only `report_body_sha256` and the runner-added top-level `volume_read_only`, recompute `hashlib.sha256(json.dumps(report, sort_keys=True, separators=(',', ':')).encode()).hexdigest()`, and compare. Do not silently rewrite report contents or coerce verifier warnings.
7. **STOP for Agent 1** to review committed verifier source/reader bridge, source-hash comparisons, report-body integrity, real Volume's 65 file digests and index/shards, ten adapters, frozen §11 values, unexpected hidden entries and logs. Agent 1 returns individual PASS/REVISE/BLOCK for delegated evidence and explicitly discloses lack of direct Modal access.

### Review attention points (do not edit approved source without fresh auditor approval)

- The verifier rejects symlinks, special files and forbidden `tokenizer.json` even in hidden folders. However, some **other hidden extra files/dirs** are recorded as `warnings.extra_hidden` rather than verifier `BLOCK`. Any such warnings must be disclosed for Agent 1 to evaluate whether the exact inventory was maintained; never silently count a `verdict=PASS` with unexpected load-relevant hidden files as full B3 acceptance.
- A verifier report's `verdict=PASS` is a **verifier-level** outcome only; `independent_pass_claimed=false` and Cursor is delegated runner. Independent **Agent 1** final PASS/REVISE/BLOCK is still required. If additional truly separate direct Volume operator access is part of the research standard, keep that constraint explicit.

## STOP boundary

**After publishing read-only results, stop for Agent 1's independent decision.** Do not pin `PINNED_CHECKPOINT_MANIFEST_SHA256` or create production authority from a candidate. Main experiment remains unauthorized; runtime lock/tiny-Gemma/B2/scientific budget are separate pending gates.
