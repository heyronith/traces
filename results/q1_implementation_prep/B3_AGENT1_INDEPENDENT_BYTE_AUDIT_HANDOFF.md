# Agent 1 — Independent B3 Byte Audit Handoff

**STOP for Agent 1 independent PASS / REVISE / BLOCK of B3 checkpoint bytes.**  
Do **not** treat the downloader’s self-check as independent.  
Scientific authorization remains **OFF**. Do **not** set `PINNED_CHECKPOINT_MANIFEST_SHA256`. Do not merge to `main`.

## What Cursor completed (not Agent 1 PASS)

| Item | Value |
|------|-------|
| Branch | `q1-impl-audit` |
| Baseline | `132179544c53f0103f8a861f1fd70b75ab4f97e0` |
| Owner hard cap (chat) | **$10 USD** |
| Modal run | https://modal.com/apps/heyronith/main/ap-IkoCtCfLZKexxLMvZsqeVh |
| Volume | `traces-q1-b3-checkpoint-bytes` |
| Secret | `hf-token` (`HF_TOKEN`) |
| Evidence dir on Volume | `/mnt/b3/evidence/2026-11-16T201645Z` |
| Candidate manifest SHA-256 (canonical bytes) | `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55` |
| Files hashed | 65 |
| Adapter-weight pairwise distinct | true |
| §11 tokenizer pins (all ten units) | matched frozen digests |
| Production pin | still **None** |
| Modal billed cost after job (workspace summary) | **$0.00** (within free credits; metered ephemeral bump negligible vs $10) |

Published redacted artifacts under `results/q1_implementation_prep/`:

- `B3_CANDIDATE_CHECKPOINT_MANIFEST.json`
- `B3_CANDIDATE_AUDIT_REPORT.json`
- `B3_CANDIDATE_SNAPSHOT_POINTER.json`
- `B3_CANDIDATE_PUBLISH_META.json`
- `B3_MODAL_DOWNLOAD_JOB_RESULT.json`
- `B3_MODAL_CPU_VERIFICATION_STATUS.md`

Code: `src/traces/q1/b3_modal_byte_verify.py`, `scripts/q1_b3_modal_byte_verify.py`  
Synthetic tests: `tests/test_q1_b3_modal_byte_verify.py`

## Frozen sources verified

- Base `google/gemma-2-2b` @ `c5ebcd40d208330abc697524c919956e692655cf`
- Adapters `interpretable-finetuning/topklora-gemma-2-2b-dense-lora` @ `6303a74fc832a34d584999364817268aefd9623b`
- Ten units: `l19/seed42–46`, `l1523/seed42–46`
- Base shards from **parsed** `model.safetensors.index.json` weight_map: `model-00001-of-00003.safetensors`, `model-00002-of-00003.safetensors`, `model-00003-of-00003.safetensors`

## Required Agent 1 actions

1. On Modal, `Volume.reload()` the same Volume snapshot; recompute SHA-256 for every materialized load-relevant file under the snapshot paths in `B3_CANDIDATE_SNAPSHOT_POINTER.json`.
2. Re-parse the base index; confirm shard inventory; confirm no symlinks inside the materialized trust roots.
3. Confirm ten adapter-weight digests pairwise distinct; all six required files per unit; forbidden `tokenizer.json` absent.
4. Re-check all four `FROZEN_SECTION11_TOKENIZER_SHA256` pins on all ten units.
5. Recompute `canonical_checkpoint_manifest_bytes` digest; compare to `d070262edae57ba4d14aae7c8f250d5999241eb6f2f56c0ad791ecfc2ff4fa55`.
6. Issue PASS / REVISE / BLOCK. **Do not** authorize production pin freeze (owner chat decision later).

## Still blocked after B3 byte audit

Owner pin freeze (B3-10), runtime/image lock, tiny-Gemma, B2/F4, hard scientific budget, model-forward / scientific execution authorization. **Q1 NOT READY FOR MODEL EXECUTION.**
