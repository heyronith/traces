# TRACES Q1 — Final Execution-Safety Corrections

**Branch:** `q1-impl-audit`  
**Commit:** `5114ebd62babfa1b94e5f9322dfa853ce9d6f30a`  
**Starting commit:** `be59764bcf8a1859af86c52dc402a57c474a3ff0`  
**Frozen science:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Protocol:** v1.3

## Verdict: engineering execution-safety PASS (not scientific readiness)

Scientific auth remains OFF by default. Production lock unavailable. F5 pending. P1–P5 remain UNRESOLVED.

## Corrections

| # | Status | Fix |
|---|--------|-----|
| 1 | CLOSED | `run_live_scientific_pipeline` loads/executes exploratory seeds 42–44 first; evaluates frozen stop; primary/secondary confirmatory callbacks only if clear; missing/invalid ranks block without fabricating defaults; spy tests prove zero confirmatory invocations |
| 2 | CLOSED | `normalize_weight_manifest_entries` maps `base:<path>` / `adapter:<path>` to filesystem paths before hash; role attribution retained; missing/wrong/unbound/ambiguous rejected; tagged pass + altered-file fail covered |
| 3 | CLOSED | `verify_runtime_allocation_ids` + live/preflight binding compare supplied ID sequences to frozen `dolly_id` order (not length-only); rejects reorder/substitute/duplicate/truncate and non-dolly instruction IDs |
| 4 | CLOSED | Per-unit `verify_unit_load_provenance` + donor/condition validation; first-unit provenance never reused for later units; complete per-unit audit evidence |

## F1 / frozen artifacts

- F1 hash preserved: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- Frozen artifact SHA baselines unchanged

## Remaining known blockers

1. **P1–P5** UNRESOLVED (original audit text unavailable)
2. **F5** owner approval pending (`F5_STATUS=OWNER_APPROVED` + env)
3. **Production runtime lock** unavailable (wheel rehash + all pins)
4. Scientific execution unauthorized (default OFF)
5. Real checkpoint load / GPU / scientific forwards not in scope

## Tests

| Suite | Result |
|-------|--------|
| Full synthetic CPU (infra + audit + reaudit + phase2 + D1–D8 + execution-safety + tiny auth-gated) | **132 passed, 12 skipped** |
| Execution-safety corrections alone | **17 passed** |
| F1 hash | preserved `fe464187…` |

**Command:**
```
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_q1_cpu_infrastructure.py \
  tests/test_q1_audit_corrections.py \
  tests/test_q1_reaudit_s1_r10.py \
  tests/test_q1_phase2_engineering.py \
  tests/test_q1_phase2_audit_corrections.py \
  tests/test_q1_final_phase2_d1_d8.py \
  tests/test_q1_execution_safety_corrections.py \
  tests/test_q1_tiny_gemma_architecture.py \
  -q --tb=line
```

Full output: `SYNTHETIC_CPU_TEST_RECORD_EXECUTION_SAFETY.txt`.
