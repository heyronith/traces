# TRACES Q1 — E2/E3 Fail-Closed Hardening (S1/S2)

**Branch:** `q1-impl-audit`  
**Baseline:** `2a27cfa21bb424e9bafae84b7deeddc73f86bc9a`  
**Protocol:** v1.3  
**Scientific auth:** OFF (default)  
**This task is not authorization for scientific execution.**

## S1 — Authorization enforcement

| Change | Detail |
|--------|--------|
| Verified tiny fixtures | `engineering_fixtures.py` — structural Gemma2 size bounds + internal marker; `mark_verified_engineering_tiny` refuses ordinary modules |
| `Q1CausalRunner` | `engineering_tiny=True` requires verified fixture at init and forward; boolean alone insufficient |
| `teacher_forced_forward` | Replaced `require_scientific_auth` bypass with mandatory `forward_authorization` (`scientific` \| `engineering_tiny`) |
| `run_f3_generation` | Removed auth bypass parameter; always requires scientific authorization |
| Tiny ≠ scientific | Tiny env flag never sets `scientific_auth_status().authorized` |

## S2 — Authenticate empirical gate evidence

| Change | Detail |
|--------|--------|
| Persisted artifacts | `empirical_evidence.py` — identity / F4 / double-BOS JSON artifacts |
| Independent hash | Claimed SHA checked against file bytes |
| Binding | Exploratory model units + frozen development ID-sequence SHA |
| Coverage | Identities 1–9; F4 families `l1523`+`l19` locked; 64 double-BOS pairs |
| Rejected | Inline `evidence_kind` / `all_passed` / `locked` / bare SHA strings; missing files; hash mismatch; wrong units; forged `all_passed` |

Fabricated local fixtures for regression only (`claims_real_experiment=false`). No real experimental evidence claimed.

## Residual blockers

- **B1–B3:** PARTIAL/BLOCKED (live runner / empirical F4–identities–double-BOS / checkpoint hashes)
- **P1–P5:** UNRESOLVED
- **F5:** pending owner approval
- **H3:** awaiting owner decision

## Verification

See `SYNTHETIC_CPU_TEST_RECORD_FAILCLOSED_S1_S2.txt`.
