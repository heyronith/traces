# TRACES Q1 — Phase 2 Internal Audit Corrections (C1–C6)

**Branch:** `q1-impl-audit`  
**Commit:** `7a8958ec6a41dd2d39a2f9e03e0f8472ff4fb90d`  
**Prior HEAD:** `01f7d869b8a0ffc671f361129f95fb57386ceca0`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`

## Verdict: **REVISE**

Corrections implemented and synthetic tests pass. Scientific execution remains **disabled**. P1–P5 still **UNRESOLVED**. F5 still pending owner approval. Production runtime lock still does not exist.

## Closure matrix

| ID | Status | Notes |
|----|--------|-------|
| C1 | CLOSED (engineering) | Typed `ScientificPipelineExecutor`; `ConditionOutputBundle` → `ModelUnitInput`; opaque callbacks fail stage5; preflight fail-closed |
| C2 | CLOSED (engineering) | F2 k=1–10 both dirs; incomplete scan refuses definitive k*; array validation; H2 exact 152+64; nonfinite≠support; H3 requires 5 seeds |
| C3 | CLOSED (engineering) | `run_study` roles: exploratory / primary / secondary / adapter-off; stop before confirmatory |
| C4 | CLOSED (engineering) | Exact layers required; manifest covers resolved files; exact revision match; PEFT disable+enable restore |
| C5 | CLOSED (engineering) | `claim_production_lock_exists` requires validated wheels, pins, intended-image metadata — flags alone insufficient |
| C6 | RETAINED UNRESOLVED | P1–P5 unchanged; no invented criteria |

## Tests (pinned env, no forwards)

| Suite | Result |
|-------|--------|
| audit corrections + phase2 + cpu + audit + reaudit | **84 passed** |
| tiny-Gemma (auth unset) | **4 passed, 12 skipped** |

## Unresolved blockers

1. P1–P5 original audit text  
2. F5 owner approval  
3. Production built-image lock  
4. Scientific authorization (OFF)  
5. Real checkpoint load verification  

Scientific-execution readiness is **not** claimed.
