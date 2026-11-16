# TRACES Q1 — B3 Modal CPU Verification Status

**Branch:** `q1-impl-audit`  
**Baseline:** `132179544c53f0103f8a861f1fd70b75ab4f97e0`  
**Owner hard cap (chat):** **$10 USD**  
**Sci auth:** OFF · **Production pin:** unset · **GPU:** none  
**Job UTC complete:** ~2026-11-16T20:19:20Z  
**Modal app:** https://modal.com/apps/heyronith/main/ap-IkoCtCfLZKexxLMvZsqeVh

## Controls

| Control | Value |
|---------|-------|
| Cap env | `B3_OWNER_APPROVED_MAX_USD=10` + `B3_CHAT_CAP_CONFIRMED=1` |
| Timeout safety-bound estimate | ~$0.24 ≪ $10 |
| CPU / mem / timeout | 2 cores / 8 GiB / 2700s |
| Volume | `traces-q1-b3-checkpoint-bytes` |
| Secret | `hf-token` (`HF_TOKEN` present) |
| Max containers | 1 |
| Workspace billed cost after job | **$0.00** |

## Downloader self-check (not independent)

| Check | Result |
|-------|--------|
| `ok` | true |
| Files hashed | 65 |
| Adapter-weight pairwise distinct | true |
| §11 pins (10×4) | all match |
| Candidate canonical SHA-256 | `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` |
| Schema validation | empty failures |
| Production pin | still None |

## Artifacts

- Code: `src/traces/q1/b3_modal_byte_verify.py`, `scripts/q1_b3_modal_byte_verify.py`
- Tests: `tests/test_q1_b3_modal_byte_verify.py` (19 passed)
- Candidate evidence: `B3_CANDIDATE_*` (not `CHECKPOINT_PROVENANCE_MANIFEST.json`)
- Handoff: `B3_AGENT1_INDEPENDENT_BYTE_AUDIT_HANDOFF.md`

## STOP

**Agent 1 independent byte audit** required next. Do not set `PINNED_CHECKPOINT_MANIFEST_SHA256`.
