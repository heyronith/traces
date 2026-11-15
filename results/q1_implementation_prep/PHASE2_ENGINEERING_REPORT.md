# TRACES Q1 — Phase 2 Engineering Report

**Branch:** `q1-impl-audit`  
**Starting commit:** `3a74c6fab41a3781cbb0096dcf83bcb1098d59e9`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Tiny-Gemma CPU validation:** preserved PASS 16/16 (`TINY_GEMMA_CPU_VALIDATION_REPORT_20261115T060416Z.md`)

## Verdict: **REVISE**

Phase 2 engineering deliverables are implemented and synthetic tests pass, but **P1–P5 remain UNRESOLVED** (original audit text unavailable), **F5 awaits owner approval**, and **production runtime lock does not exist**. Do **not** treat this as scientific execution readiness.

---

## 1. Commit SHA

*(filled at publish)*

## 2. P1–P5 traceability

See `P1_P5_TRACEABILITY_MATRIX.md`.

| ID | Status |
|----|--------|
| P1–P5 | **UNRESOLVED** — original independent-audit definitions not located; no invented closure |

## 3. Engineering changes and tests

### Statistical pipeline
- F2: joint top-k evaluations via actual `joint_set_effects_*` arrays; missing joints → indeterminate (no head-sum proxy)
- Fold A/B selection with allocation-order membership; k=1–10 both directions; fold-specific bootstrap CIs with fixed selections
- H2: upper 90% bounds for every upstream head + all 64 pairs (when route-denom OK)
- H3: actual top-2 sets across primary seeds in `run_study`
- Route-denominator gates precede route-relative share conclusions
- Empty eligible / invalid denom → fail-closed or indeterminate
- F1 RNG/hash preserved: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`

### Connected driver
- `scientific_execution.py`: 8-stage plan; `TRACES_Q1_ALLOW_SCIENTIFIC_EXECUTION` (default OFF); independent of tiny-Gemma auth
- Analysis-only path for precomputed effects; live path refuses without sci auth

### Loader
- Weight SHA mandatory; dtype/attn fail-closed; adapter-off verification; policy helpers for mocks

### Tests (pinned env, no forwards)
| Suite | Result |
|-------|--------|
| phase2 + cpu + audit + reaudit | **62 passed** |
| tiny-Gemma (auth unset) | **4 passed, 12 skipped** |
| Scientific / pretrained / GPU | **not run** |

## 4. Pipeline coverage

`CONNECTED_PIPELINE_COVERAGE.md` — stages 1–8 documented; live forwards not invoked.

## 5. Loader provenance

`LOADER_PROVENANCE_REPORT.md`

## 6. F5 owner decision

`F5_OWNER_DECISION_NOTE.md` — **awaiting explicit owner approval** (no approval claimed).

## 7. Runtime lock

| Item | Status |
|------|--------|
| Procedure + `runtime_lock.py` | Ready |
| Synthetic fixture lock | Generated & validated (`runtime_lock_synthetic_fixture.json`) |
| Production built-image lock | **DOES NOT EXIST** |

## 8. Unresolved blockers

1. **P1–P5** — original finding text unavailable  
2. **F5** — owner approval pending  
3. **Production runtime lock** — not generated from intended image  
4. **Real loader verification** on pinned scientific weights — not authorized  
5. **Scientific execution authorization** — OFF; no scientific forwards  
6. Independent audit of this engineering commit still required  

## 9. Final verdict

**REVISE** — engineering advanced and audit-ready for review of code/tests, but not READY FOR INDEPENDENT AUDIT as a complete scientific-execution package while P1–P5/F5/production lock remain open. Not BLOCK solely on synthetic failures (there were none).
