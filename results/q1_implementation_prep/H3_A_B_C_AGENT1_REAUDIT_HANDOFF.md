# Agent 1 — Targeted Independent Re-Audit Handoff (H3-A / H3-B / H3-C)

**STOP for Agent 1 independent review of H3-A, H3-B, and H3-C only.**  
Do not merge to `main`. Scientific authorization remains OFF.  
**Do not treat Cursor suite results as an independent PASS.**

## Scope

Correct the three Agent 1 aggregation findings per `H3_A_B_OWNER_APPROVAL_RECORD_20261115.md`.

Preserve accepted per-seed applicable-direction H3-B, and **B1, R1, M1, D2–D4, M2, S2, D5**. Reopen only with a new reproducible regression.

Matrix: `results/q1_implementation_prep/H3_A_B_C_CORRECTION_MATRIX.md`  
Suite record: `results/q1_implementation_prep/SYNTHETIC_CPU_TEST_RECORD_H3_A_B_C.txt`

## Exact trees

| Role | SHA |
|------|-----|
| Baseline | `b48058839e5a57fbe43235a670a15cf92977c9f4` |
| Prior H3-B code | `5fd62ff33f09b93b113996e778bd425f2bad50fb` |
| Correction commit (code) | `3e44e950c2ba3ee6edc94b427c203e7f6cd8ccd1` |
| Correction tree | `9b78ba6d77890f43a672cbc94e63e7c3a1a67748` |
| Frozen main | `b1e8874f45311678e6fa241f504e7e09d4a7808c` |
| Publish tip | `origin/q1-impl-audit` HEAD |

## Cursor-reported verification (not Agent 1 PASS)

- New: `tests/test_q1_h3_a_b_c_corrections.py`
- Updated: `tests/test_q1_h3_applicable_direction.py`, `tests/test_q1_phase2_engineering.py`
- Full permitted Q1 synthetic suite: **371 passed, 12 skipped**
- Env: CPython **3.11.15**, NumPy **2.1.3**, torch **2.5.1**, transformers **4.51.3**, PEFT **0.19.1**
- F1: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- Pin remains **unset**

## Suggested re-audit focus (PASS / REVISE / BLOCK each)

1. **H3-A:** 4+1 conserved → H3a supported, H3b **not**; 3+2 vary → H3b only; no dual support.
2. **H3-B:** endpoint statuses; §5.4(3) on H3 endpoint (route INCONCLUSIVE **or** ≥2 gate_failed/h3_indeterminate); 1 indeterminate + 4 conserved may still H3a; 2 determinate non-support ≠ false INCONCLUSIVE; R-moved disagreement determinate.
3. **H3-C:** exact `primary_l19_seed42…46` required for route + H3; arbitrary/renamed/duplicate/missing refuse; valid fixed roster with one inapplicable keeps identity.
4. **Regressions:** per-seed R-and/R-or/R-moved; F1; B1/S2/D5.

## Outstanding (NOT READY)

B3 byte provenance / owner pin, runtime lock, B2/F4/identities, tiny-Gemma auth, budget, separate scientific execution authorization.
