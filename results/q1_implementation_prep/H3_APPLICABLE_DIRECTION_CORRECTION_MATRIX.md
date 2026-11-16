# H3 Applicable-Direction (H3-B) Correction Matrix

**Branch:** `q1-impl-audit`  
**Baseline tip:** `0adfcc1157c6bb56bd8389a05bbaf4da3c57e35e`  
**Correction commit (code):** `5fd62ff33f09b93b113996e778bd425f2bad50fb`  
**Correction tree:** `4c28ec7937e32f232814a51c28a9965b5994985f`  
**Suite:** 351 passed, 12 skipped  
**F1:** `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Owner record:** `F5_H3_OWNER_APPROVAL_INTERPRETATION_RECORD_20261115.md`  
**Scientific authorization:** OFF · **Owner pin:** unset

| Item | Before | After |
|------|--------|-------|
| Direction selection | Forward-only `head_effects_fwd` pre-map for all five primary seeds | Per-seed applicable direction after gates/route/H1/F2 |
| R-and | Forward (coincidentally) | Signed upstream **forward** ranking only |
| R-or | Forward (incorrect) | Signed upstream **reverse** ranking only |
| R-moved | Forward-only | Both directions; full support only if both `k*≤2`, complete rankings, denoms OK, **identical** unordered top-2; else directional/partial |
| H1 | Mixed with group map | Explicit **H1-fail** required; H1-indeterminate/pass never support |
| Study H3a/H3b | Required exactly five valid top-2 sets; varies could include nonqualifying | Fixed five-identity roster; ≥4 individually qualifying; inapplicable = non-supporting; H3b varies **only** among qualifying sets |
| §5.4(3) | Study INCONCLUSIVE separate | INCONCLUSIVE **blocks** full H3 claims (precedence) |
| Group pre-map | Built before unit analysis | Removed; `top2_by_seed` arg ignored for H3 direction |

## Primary files

- `src/traces/q1/analysis_ext.py` — `assess_unit_h3_direction_support`, `h3_study_level_from_seed_records`, updated `h3_top2_conservation`
- `src/traces/q1/driver.py` — per-unit H3 evidence; study aggregation in `run_study`
- `tests/test_q1_h3_applicable_direction.py` — new focused suite
- `tests/test_q1_phase2_engineering.py` — driver output contract update

## Preserved

B1, R1, M1, D2–D4, M2, S2, D5 auth; F1 stream; frozen protocol files; F5 BF16 path; exploratory/secondary/adapter-off roles unchanged.

## Explicit non-claims

Not scientific readiness. B3 pin, runtime lock, tiny-Gemma, B2 evidence, budget, and sci-auth remain **NOT READY**.
