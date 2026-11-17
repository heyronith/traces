# Stage B — Agent 1 runtime-lock identity REVISE remediation

**State:** Focused security code/test fix committed for validation; **owner runtime lock SHA pin deliberately UNSET** until explicit approval. Production/runtime-science gate remains fail-closed. No Modal jobs, GPU, downloads, models, or merge authorized by this record.

**Agent 1 finding (2026-11-17):** the real runtime lock's immutable image identity is not owner-pinned; the prior production claim could be true for a substituted image digest/tag or an internally consistent CUDA 12.4 wheel set. Scientific preflight accepted a caller lock path without validating `MODAL_IMAGE_ID` of the running container.

## Scope of the correction

- `src/traces/q1/runtime_lock.py`:
  - Adds `PINNED_RUNTIME_LOCK_SHA256: str | None = None` pending direct owner freeze. Neither a code commit nor a caller lock file can self-authorize the pin.
  - Adds `runtime_lock_matches_owner_pin(doc, lock_path=...)` to hash the exact **raw lock bytes** against the owner constant and require the in-memory mapping equal the parsed file. `claim_production_lock_exists` requires this proof, exact wheel rehash, and existing pin/CUDA/inventory checks.
  - Disallows a caller-provided alternate image mapping when the pinned lock is checked.
  - Adds `running_modal_image_matches_lock` to require nonempty, equal `MODAL_IMAGE_ID` in the real executor environment; no caller override or silent missing-ID fallback.
- `src/traces/q1/scientific_execution.py`: production preflight passes the lock path into pinned-byte validation, requires actual running Modal image ID to match the pinned lock image, and checks installed direct dependency/Python versions before allowing `production_lock.claim=true`. No model callbacks are invoked by preflight.
- `tests/test_q1_stage_b_owner_runtime_identity_pin.py`: CPU-only tests covering owner-pin unset, positive exact pinned raw bytes via **test-local monkeypatch**, missing path, in-memory tampering, changed image ID, base digest, image tag, coherent different CUDA-version wheel set, actual `MODAL_IMAGE_ID` missing/wrong/matching, and preflight integration. Synthetic test fixture is never accepted as production without explicit test-local mocked owner pin.

## Approved candidate to freeze after focused testing/review

- Lock: `results/q1_implementation_prep/TRACES_Q1_RUNTIME_LOCK_BUILT_IMAGE_CORRECTED.json`
- Audited candidate raw SHA-256: **`f44cd2095d6a666fd33481f5f25636fed677914a5a8ebcb92f463bcd0a716a72`**.
- Intended Modal image ID embedded in candidate: **`im-0j3j9DzkkZSLdRQTxl4scU`**.
- Corrected Stage B lock and associated earlier wheel evidence remain unchanged. `main` remains frozen at `b1e8874f45311678e6fa241f504e7e09d4a7808c`.

## Cursor's one scoped CPU-only validation

Use pinned CPython 3.11.15 and frozen dependencies, with `TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION` and `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD` unset. Run:

```bash
python -m pytest \
  tests/test_q1_stage_b_owner_runtime_identity_pin.py \
  tests/test_q1_stage_b_runtime_lock_remediation.py \
  tests/test_q1_b3_runtime_provenance.py \
  tests/test_q1_cpu_infrastructure.py \
  -q --tb=short
```

Verify current owner pin stays None; exercise both positive test-local pin simulation and fail-closed negatives. Record exact commit/tree, passed/failed counts, environment and unchanged candidate lock SHA. Do not run the Modal correction script, rebuild an image or incur cloud costs. Return PASS/REVISE/BLOCK for **this fix only**.

## Decision after successful validation

**Owner explicit action required:** approve the above exact raw lock SHA-256 for `PINNED_RUNTIME_LOCK_SHA256`. Owner approval should be distinct from code/test review, image-build authorization or Stage C science permission. Then set the literal owner-approved SHA **once on `q1-impl-audit`**, run focused post-freeze CPU regressions, obtain Agent 1 focused confirmation of the freeze and correct runtime `MODAL_IMAGE_ID` enforcement, and close Stage B. Do not edit or re-serialize the pinned lock file. No model execution or merge to `main`.

**Stage C** real-model exploratory calibration remains separately unauthorized, even after Stage B closes.
