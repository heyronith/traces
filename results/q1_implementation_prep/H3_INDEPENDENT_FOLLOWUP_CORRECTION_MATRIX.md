# H3 Independent Follow-up Verification Matrix

**Branch:** `q1-impl-audit`  
**Starting commit:** `08c75a2a76cf5abd68915f449e34d968a421a249`  
**Validation correction:** `880017a47b198c79dbd64693aa3521528818811d`  
**Follow-up tests:** `382303f721866a15b8475b9ddb1aba0acd459031`  
**Code tree (at start tip):** `13fd6b5f89e8799a58dccccc41e3a3b9b6d25b06`  
**Clean tested tip:** `a80ca32401288743b5d59b266cb739ae5904742b`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Suite (this verification):** 403 passed, 12 skipped  
**F1:** `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`  
**Pin:** unset · **Sci auth:** OFF  
**Engineering edits this session:** none (verification-only)

| Claim | Result |
|-------|--------|
| Contradictory status/eligibility → `VALIDATION_BLOCKED` | PASS (helper + `run_study` boundary) |
| Invalid / non-upstream top-2 head IDs cannot support H3 | PASS |
| Genuine synthetic H3a/H3b via real `run_model_unit_analysis` + `run_study` (no mock of either on positive path) | PASS |
| H3-A mutual exclusion intact | PASS |
| H3-B decisive-over-uncertain intact | PASS |
| Strict `json.dumps(..., allow_nan=False)` without `default=str` | PASS |
| F1 reference unchanged | PASS |

## Files reviewed

- `src/traces/q1/analysis_ext.py` — preserve contradiction for validation; require exact int upstream heads in top-2; technical_invalid counting
- `src/traces/q1/driver.py` — do not coerce contradictory eligibility; only admit eligible top-2 when status/full/upstream-valid
- `tests/test_q1_h3_independent_followup.py` — 10 new regressions + genuine positive paths

## Explicit non-claims

Not scientific readiness. Cursor suite is **not** independent Agent 1 PASS. B3 pin, runtime lock, tiny-Gemma, B2, budget, sci-auth remain **NOT READY**.
