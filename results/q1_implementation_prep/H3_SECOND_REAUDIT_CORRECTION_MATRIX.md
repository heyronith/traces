# H3 Second Re-Audit Targeted Correction Matrix

**Branch:** `q1-impl-audit`  
**Baseline tip:** `164ada84032f97f195d973023badaacd0e279f64`  
**Correction commit (code):** `894e5258207355961d353cbe6e21734f159e264d`  
**Correction tree:** `f9c80951e1594d017a42808beec4ae9ca2680fb3`  
**Clean tested tip:** `5724f0dec337f0a0bc9d613927fdbe9158968365`  
**Prior H3-A/B/C code:** `3e44e950c2ba3ee6edc94b427c203e7f6cd8ccd1` / tree `9b78ba6d77890f43a672cbc94e63e7c3a1a67748`  
**Owner approval:** `H3_DECISIVE_OVER_UNCERTAIN_OWNER_APPROVAL_20261116.md`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Suite:** 393 passed, 12 skipped  
**Scientific authorization:** OFF · **Owner pin:** unset  
**F1:** `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`

| ID | Defect | Correction | Primary files |
|----|--------|------------|---------------|
| **H3-B** | Uncertainty could outrank definitive R-moved disagreement / k*>10 | Decisive-over-uncertain: complete top-2 disagreement and definitive k*>10 → `determinate_non_support` before H1-indeterminate / undetermined k*; preserve uncertainty payload | `analysis_ext.py` `assess_unit_h3_direction_support` |
| **H3-B** | Unknown route (`R-weird`) treated as scientific non-support | Raise `ValueError` / record `technical_invalid`; driver never rewrites to `determinate_non_support` | `analysis_ext.py`, `driver.py` |
| **H3-B** | Contradictory `full_qualifying` + ineligible counted as qualifying | Validation-blocked study; no H3a/H3b from inconsistent technical records | `analysis_ext.py` `h3_study_level_from_seed_records` |
| **H3-C** | Identity-only roster check | Require exact `primary_l19_seed42…46` **and** each `family=="l19"`; spoof family refuses route+H3 | `driver.py` `run_study` |
| **Coverage** | No true `run_study` H3a/H3b success path | Two e2e tests through real `run_study` + aggregator; family-spoof + inconclusive adverse | `tests/test_q1_h3_second_reaudit.py` |

## Preserved

H3-A mutual exclusion (4+1 → H3a only; 3+2 → H3b only); applicable-direction R-and/R-or/R-moved; F1/F2; B1/S2/D5; exploratory/secondary/adapter-off; frozen protocol; main.

## Explicit non-claims

Not scientific readiness. Cursor suite is **not** independent PASS. B3 pin, runtime lock, tiny-Gemma, B2, budget, sci-auth remain **NOT READY**.
