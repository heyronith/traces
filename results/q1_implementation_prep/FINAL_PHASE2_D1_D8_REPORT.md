# TRACES Q1 — Final Phase 2 Internal Corrections (D1–D8)

**Branch:** `q1-impl-audit`  
**Commit:** `f8fd73ea47c7b9e84f7561bdf77f48ffc86f6425`  
**Prior HEAD:** `4dab7c2aa2acd703b8c0b82ee3d2be335b678404`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`

## Verdict: **PASS** (engineering audit preparation only)

Not scientific-execution readiness. Scientific auth remains OFF. Production lock remains unavailable. P1–P5 remain UNRESOLVED. F5 remains pending.

## Closure

| ID | Status | Fix |
|----|--------|-----|
| D1 | CLOSED | Strict `ConditionOutputBundle.validate()` — eligible-only lengths, finite values, mask/ID ordering; rejects full-denominator misuse |
| D2 | CLOSED | `verify_frozen_artifacts()` compares working-tree SHA256 + ID-sequence hashes to immutable freeze baselines |
| D3 | CLOSED | Production claim requires on-disk wheel rehash, all direct-pin exact versions, intended-image metadata; synthetic fixtures never suffice |
| D4 | CLOSED | Require exact l1523 seeds 42–44 + three ranks; no confirmatory when absent; stop on ≥2/3 rank>10; no fabricated defaults |
| D5 | CLOSED | R-and→forward H1/k*/H2; R-or→reverse; R-moved→bidirectional; other routes inapplicable |
| D6 | CLOSED | Manifest must use `base:`/`adapter:` tags or exact resolved inventories; unbound substring names rejected |
| D7 | CLOSED | Multi-unit live path with Q1CausalRunner adapters; reconcile boundary/intervention; frozen unit set |
| D8 | CLOSED | Exact `F5_STATUS=OWNER_APPROVED` + signer + env; `NOT APPROVED` does not authorize |
| P1–P5 | UNRESOLVED | Retained; no invented definitions |

## Tests

| Suite | Result |
|-------|--------|
| D1–D8 + prior phase2 + cpu + audit + reaudit | **111 passed** |
| tiny-Gemma (auth unset) | **4 passed, 12 skipped** |
| F1 hash | preserved `fe464187…` |

## Remaining known defects / blockers

1. P1–P5 original audit text unavailable  
2. F5 owner approval pending  
3. Production built-image lock does not exist  
4. Scientific execution unauthorized (default OFF)  
5. Real checkpoint load / GPU not in scope  

Tiny-Gemma 16/16 PASS evidence preserved from prior authorized run.
