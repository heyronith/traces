# Runtime-lock preparation status

**Production built-image lock:** **NOT CLAIMED** (Stage B BLOCK/INCOMPLETE — see `STAGE_B_RUNTIME_LOCK_STATUS.md`)

| Artifact | Status |
|----------|--------|
| `src/traces/q1/runtime_lock.py` | Implemented (generate / validate / production claim gate) |
| `scripts/q1_prepare_runtime_lock.py` | Extended with `--synthetic-wheel-dir` |
| Stage A prep | `production_image_lock_exists=false` |
| Stage B Modal image + 41 wheel hashes | Captured; `validate_lock_document` ok |
| `claim_production_lock_exists` | **false** (torch wheel version `2.5.1+cu121` ≠ pin string `2.5.1`) |
| A10G identity probe | Succeeded (no model load) |

Do not claim a production image lock until `claim_production_lock_exists(..., require_rehash=True)` returns true against real wheel evidence.
