# H3-A / H3-B / H3-C Correction Matrix

**Branch:** `q1-impl-audit`  
**Baseline tip:** `b48058839e5a57fbe43235a670a15cf92977c9f4`  
**Correction commit (code):** `3e44e950c2ba3ee6edc94b427c203e7f6cd8ccd1`  
**Correction tree:** `9b78ba6d77890f43a672cbc94e63e7c3a1a67748`  
**Suite:** 371 passed, 12 skipped  
**F1:** `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`  
**Prior H3-B code:** `5fd62ff33f09b93b113996e778bd425f2bad50fb` / tree `4c28ec7937e32f232814a51c28a9965b5994985f`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Owner approval:** `H3_A_B_OWNER_APPROVAL_RECORD_20261115.md`  
**Scientific authorization:** OFF · **Owner pin:** unset

| ID | Defect | Correction | Primary files |
|----|--------|------------|---------------|
| **H3-A** | Dual H3a+H3b support on 4+1 conservation | Mutually exclusive: ≥4/5 same unordered pair → **H3a only**; ≥4 qualify with no 4/5 pair → **H3b only**; minority pairs disclosed descriptively | `analysis_ext.py` `h3_study_level_from_seed_records` |
| **H3-B** | Only route-level INCONCLUSIVE blocked H3; H1-indeterminate / undetermined k* did not | Per-seed `h3_endpoint_status`: `full_qualifying` / `h3_indeterminate` / `determinate_non_support` / `gate_failed`; H3 INCONCLUSIVE if route INCONCLUSIVE **or** gate_failed+h3_indeterminate ≥2 | `analysis_ext.py` `assess_unit_h3_direction_support`, study aggregator; `driver.py` |
| **H3-C** | Fallback to arbitrary observed five IDs | Require exact `{primary_l19_seed42…46}`; refuse canonical route + H3 on mismatch/duplicate/missing/rename | `driver.py` `run_study` |

## Preserved

Per-seed R-and/R-or/R-moved direction, H1-fail, F1/F2, 152 upstream, B1/S2/D5, F5 BF16, exploratory/secondary/adapter-off roles, frozen protocol bytes.

## Explicit non-claims

Not scientific readiness. B3 pin, runtime lock, tiny-Gemma, B2, budget, sci-auth remain **NOT READY**. Independent PASS is for Agent 1 only.
