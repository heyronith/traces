# Agent 1 — Targeted Independent Re-Audit Handoff (H3 Applicable Direction)

**STOP for Agent 1 independent review of H3-B only.**  
Do not merge to `main`. Scientific authorization remains OFF.  
**Do not treat Cursor suite results as an independent PASS.**

## Scope

Implement and verify the owner-approved applicable-direction H3 interpretation only (`F5_H3_OWNER_APPROVAL_INTERPRETATION_RECORD_20261115.md`).

Preserve accepted **B1, R1, M1, D2–D4, M2, S2, D5**. Do not reopen without a new reproducible regression. Do not amend frozen protocol files or change F1/F2 splits.

Matrix: `results/q1_implementation_prep/H3_APPLICABLE_DIRECTION_CORRECTION_MATRIX.md`  
Suite record: `results/q1_implementation_prep/SYNTHETIC_CPU_TEST_RECORD_H3_APPLICABLE_DIRECTION.txt`

## Exact trees

| Role | SHA |
|------|-----|
| Baseline | `0adfcc1157c6bb56bd8389a05bbaf4da3c57e35e` |
| Correction commit (code) | `5fd62ff33f09b93b113996e778bd425f2bad50fb` |
| Correction tree | `4c28ec7937e32f232814a51c28a9965b5994985f` |
| Frozen main | `b1e8874f45311678e6fa241f504e7e09d4a7808c` |
| Clean tested tip (suite + handoff) | `99c1e5ac143a2c137151c8d2a2114fcda21b412c` |
| Publish tip | `origin/q1-impl-audit` HEAD |

## Cursor-reported verification (engineering only — not Agent 1 PASS)

- New tests: `tests/test_q1_h3_applicable_direction.py`
- Updated: `tests/test_q1_phase2_engineering.py` (driver H3 output contract)
- Full permitted Q1 synthetic suite: **351 passed, 12 skipped**
- Environment: CPython **3.11.15**, NumPy **2.1.3**, torch **2.5.1**, transformers **4.51.3**, PEFT **0.19.1**
- F1: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- `PINNED_CHECKPOINT_MANIFEST_SHA256` remains **unset**

## Suggested re-audit focus (PASS / REVISE / BLOCK)

1. **Per-seed direction:** R-and→forward; R-or→reverse; mixed cohort uses each seed’s applicable direction (no favorable-direction selection).
2. **R-moved:** both directions + both `k*≤2` + identical unordered top-2 for full support; disagreement / incomplete ranking / denom failure → non-supporting with both directions reported.
3. **H1:** explicit H1-fail required; H1-indeterminate / H1-pass never full H3 support.
4. **Four-of-five:** fixed five primary identities; ≥4 individually qualifying; inapplicable fifth does not erase four; does not create spurious H3b variation; three qualifying → no full claim.
5. **H3a/H3b aggregation:** conserved pair vs varied qualifying sets; dual reporting without invented precedence.
6. **§5.4(3):** ≥2/5 gate-failed or Indeterminate → study INCONCLUSIVE blocks full H3 claims.
7. **Regressions:** accepted B1/S2/D5 surfaces; F1 hash; no forward-only group pre-map.

## Outstanding (NOT READY)

Real B3 byte provenance / owner pin, built-image runtime lock, tiny-Gemma revalidation, B2 F4/identities/double-BOS, hard budget, and **separate scientific execution authorization**.

## Explicit non-claims

No pretrained loads, downloads, GPU, scientific forwards, expenditure, protocol changes, or merge to `main`. Independent PASS is for Agent 1 only after reproduction.
