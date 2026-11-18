# B2 exploratory CPU preparation — identities 1–9 + descriptive double-BOS

**Verdict: CPU_PREP_READY** (2026-11-17). No GPU inference in this step. Preserves published F4. Does not alter the single-BOS exploratory stopping rule. No held-out / confirmatory / main.

## F4 publication discrepancy (resolved)

| Check | Result |
| --- | --- |
| Local HEAD before push | `3fea2c2e27fa0bd5ff9fb70f5319a7197b811327` (ahead 1 of `953e6a8…`) |
| Push | `git push origin HEAD:q1-impl-audit` (no force; no fetch/pull) → `953e6a8..3fea2c2` |
| Remote `refs/heads/q1-impl-audit` | `3fea2c2e27fa0bd5ff9fb70f5319a7197b811327` |
| Tree | `165b6de6ded428fcb9f129c57a85c6a7ce3117b8` |
| Report present on remote | `F4_EXPLORATORY_FAMILY_CALIBRATION_REPORT_20261117.md` |
| Artifact SHA-256 (remote bytes) | `80d7985c3d6241c5661cfacccbea2764c0dec429a9f637131d531727de41d021` |
| 900 forwards re-run? | **No** |

## CPU prep delivered

- Module: `src/traces/q1/b2_exploratory_cpu_prep.py`
- Tests: `tests/test_q1_b2_exploratory_cpu_prep.py` → **8/8 PASS** (CPython 3.11.15; science flags unset)
- Package: `results/q1_implementation_prep/b2_cpu_prep/B2_EXPLORATORY_CPU_PREP_PACKAGE.json`
- Bound F4 epsilons (all families): **0.001**
- Algebraic synthetic identity suite 1–9 with locked F4 epsilons: **all_passed**
- Double-BOS catalog: **64** forward L15×L16 pairs; descriptive only
- Synthetic identity / double-BOS fixtures: schema-exercise only; production auth **blocked** as required

## Planning estimates (not GPU authorization)

| Gate | Approx forwards | Est. A10G USD | Plan cap |
| --- | ---: | ---: | ---: |
| Identities 1–9 (order-of-magnitude) | intact + ~12 intervention arms/prompt + gen | ~$0.28 | $3 |
| Double-BOS 64-pair descriptive | 300 intact + 9,600 patched = 9,900 | ~$0.60 | $5 |

Dedicated GPU runners are still required before any real-model identity or double-BOS collection. Standing non-main authorization covers those runs once runners exist; this package does **not** launch them.

## Explicit stop rules preserved

No held-out data, confirmatory experiment, main merge, F4 overwrite, or change to the single-BOS exploratory stopping rule. Next GPU work: dedicated identity runner, then dedicated double-BOS runner, each with fail-closed metering.
