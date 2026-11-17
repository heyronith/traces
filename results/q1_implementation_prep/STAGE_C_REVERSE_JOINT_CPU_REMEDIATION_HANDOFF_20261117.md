# Stage C first real Gemma — reverse-joint REVISE and CPU-only remediation handoff

**Status:** Stage C first run **REVISE / STOP**. This document authorizes **NO** additional Modal/GPU/model use or spending. First bounded Stage C session authorization has been consumed, including aborted attempts; a second session requires a **new explicit owner decision**.

## Published first-run ground truth

First-run approval baseline: commit `102015fee9906f97f696430ab7f4da4b0f524461`, tree `b6fe0ea749be6d82956d4dd9ee86aa692583bb00`. Evidence publication: commit `e3aa5b60f00ff96acbd71050fe5a9ef9cde38afd`, tree `e9edb13f4a141975ea111ca0d43918f81e180cd1`.

- `STAGE_C_FIRST_REAL_GEMMA_EXPLORATORY_REPORT.md`: original `REVISE` report, **not overwritten**.
- `STAGE_C_FIRST_REAL_GEMMA_EXPLORATORY_EVIDENCE.json`: 172 timing rows, **148 ok and 24 fail**; pinned image, 54-wheel/runtime identity, model load, BOS/tokenizer/dtype/eager passed.
- `STAGE_C_FIRST_REAL_GEMMA_FAILURES_REDACTED.json`: all 24 failure rows in exactly two reverse-joint arms, `all_upstream_joint_reverse` and `eight_head_layer_joint_reverse` (4 frozen dev prompts × 3 repeats × 2 arms), error `Recipient tag identity mismatch: prompt=T direction=reverse expects=C`.
- No held-out data, no production scientific pipeline, no F4 family table, no Stage D. Stage C metered cost reported **~$0.055 / $3** previously approved for **the consumed first session**, not rolled forward to another.
- Confirmed source root cause in `scripts/q1_stage_c_modal_first_real_gemma.py`: both reverse joint loops passed **T** input IDs, mask, expected-full token IDs, and tag. A tag-only fix is **incorrect**; it must switch all four to **C**.

## CPU-only correction (unexecuted)

- `src/traces/q1/stage_c_exploratory.py`: new pure selector `select_stage_c_joint_recipient(pair, direction)` binds **forward=T and reverse=C** tokens, attention mask, tag and full-sequence expected IDs; rejects invalid directions, missing fields and non-batch-one/misaligned sequences.
- `scripts/q1_stage_c_modal_first_real_gemma.py`: both joint arm sites use the selector for all four recipient inputs; the single-head and other **successful** arms remain untouched.
- **Both** the old local `main()` and old remote `run_stage_c_calibration()` entrypoints fail closed while `FIRST_STAGE_C_SESSION_CLOSED=True`. Never flip this constant to replay all 172 arms; only a separately authorized dedicated **reverse-only** run may be considered later.
- `tests/test_q1_stage_c_reverse_joint_recipient.py`: CPU token-tensor tests for direction binding and bad inputs, plus static check that both joint sites use the selector and that the consumed first-session entrypoints stay blocked.

## Cursor task — local CPU only

1. Check out the current audited `q1-impl-audit` HEAD; report exact commit and tree. Use pinned CPython 3.11.15. Keep `TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION` and `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD` **unset**, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. No GPU, Modal calls, model loads, checkpoints or output re-evaluation.
2. Run:

```bash
python -m pytest \
  tests/test_q1_stage_c_reverse_joint_recipient.py \
  tests/test_q1_stage_c_exploratory_guards.py \
  tests/test_q1_pre_stage_c_running_inventory.py \
  tests/test_q1_stage_b_owner_runtime_identity_pin.py \
  -q --tb=short
```

3. Confirm explicitly both **C recipient tokens and C tag** for reverse arms and both **T tokens and T tag** for forward arms; full expected ID proof and mask match the chosen tokens; bad directions/masks fail; original `run_head_patches` recipient verification is unchanged. Confirm full-session entrypoints blocked. If a test fails, correct only this limited source/test scope and rerun CPU tests.
4. Commit any focused test correction and publish **PASS / REVISE / BLOCK** with tested commit/tree and counts. **STOP** without new GPU spend.

## Next separate owner decision (not yet requested or granted)

Only after CPU tests PASS, request approval for a **reverse-only** A10/A10G calibration on the same one checkpoint and same four development prompts, no forward or already-passed arms. Proposed **new session cap**: **$1.00 cumulative, 15 minutes total**, subject to owner decision. Implement a dedicated runner with hard-fail direction/arm whitelist and current image/checkpoint/runtime locks before launching; independent running-image/BOS identities, one warmup as needed, only the previously missing 24 reverse arm rows, accurate separate costs. No full 172-arm replay, no scientific claim/held-out/model expansion. No owner approval is inferred from this CPU remediation record.

Frozen `main` `b1e8874f45311678e6fa241f504e7e09d4a7808c` remains unchanged.
