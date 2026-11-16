# TRACES Q1 — Agent 1 H3 Re-Audit Follow-up Remediation

**Disposition:** CODE CHANGES PUBLISHED — **INDEPENDENT TEST PASS NOT CLAIMED**. Further validation required before declaring H3 PASS.  
**Scientific execution authorization:** OFF. No merge to main, pretrained model load, model forward, GPU, network checkpoint download or expenditure.

## Anchors

- Frozen main: `b1e8874f45311678e6fa241f504e7e09d4a7808c` (verified unchanged)
- Previous publish tip: `808aef478da1ffd0d27c21e3e65452dc96667560`
- Validation correction commit: `880017a47b198c79dbd64693aa3521528818811d`
- Added regression / full-path synthetic test commit: `382303f721866a15b8475b9ddb1aba0acd459031`
- Branch: `q1-impl-audit`
- Code/test changes only: `src/traces/q1/analysis_ext.py`, `src/traces/q1/driver.py`, `tests/test_q1_h3_independent_followup.py`

## Corrections

1. H3-B: preserve unmodified `full_h3_eligible` and status through `run_study`. A contradiction must reach `_validate_h3_seed_record_consistency`, not be normalized into an apparently valid scientific endpoint.
2. H3-B: require each top-2 identity to be a genuine member of `EXPECTED_UPSTREAM_HEADS`, with exact integer layer/head indices. Malformed, out-of-range and boolean-substituted heads must validation-block.
3. H3-B: invalid records count as `technical_invalid`, never as `full_qualifying` or scientific non-support.
4. H3a/H3b integration: new parameterized synthetic tests use real `run_model_unit_analysis` through `run_study` and actual H3 aggregation. Inputs are complete synthetic upstream-head maps, 64 pair effects, 200 held-out IDs, 50 dev IDs, and synthetic precomputed jointly patched F2 k=1..10 measurements (not a sum proxy). No model execution.
5. New tests check true H3a/H3b positive paths, contradictory records at both helper and driver boundaries, invalid head IDs, unknown explicit statuses, and **strict** `json.dumps(payload, allow_nan=False)` with **no** `default=str`.

## Pending verification (not fabricated)

The earlier `393 passed, 12 skipped` result applies to the *previous* code, not these commits. This assistant's isolated Python environment is 3.13, not the pinned Python 3.11.15 venv, and the repository cannot be checked out locally because outgoing GitHub DNS/network access is unavailable. No independent pinned-suite result is asserted. No GitHub CI runs were present for the new test commit at review time.

On an independently controlled pinned local environment, run:

```bash
PYTHONPATH=src:tests results/q1_final_cpu_checks/.venv_tf4513/bin/python -m pytest \
  tests/test_q1_h3_independent_followup.py \
  tests/test_q1_h3_second_reaudit.py \
  tests/test_q1_h3_a_b_c_corrections.py \
  tests/test_q1_h3_applicable_direction.py \
  -q --tb=line
```

Then rerun the **entire permitted** suite from `SYNTHETIC_CPU_TEST_RECORD_H3_SECOND_REAUDIT.txt`, additionally including `tests/test_q1_h3_independent_followup.py`. Record exact environment, command, exit code and results on clean committed code; recheck F1 SHA `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`. The owner manifest pin remains unset.

## Outstanding blockers

B3 real checkpoint byte verification + owner manifest SHA pin, production image/wheel/CUDA lock, independently approved tiny-Gemma revalidation, B2 F4/identity/double-BOS evidence, workload calibration, hard budget, and separate scientific execution authorization. All **NOT READY**.
