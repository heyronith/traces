# TRACES Q1 — Final Execution-Safety Fixes (round 2)

**Branch:** `q1-impl-audit`  
**Starting HEAD:** `b711b40042c7b33d101053de31db2f2c8149b97d`  
**Frozen science:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Protocol:** v1.3

## Verdict: engineering execution-safety PASS (not scientific readiness)

Scientific auth remains OFF. Production lock unavailable. F5 pending. P1–P5 remain UNRESOLVED.

## Per-defect closure

| # | Status | Evidence |
|---|--------|----------|
| 1 | CLOSED | Live gate computes nominated-pair ranks from completed exploratory intervention maps (l1523 seeds 42–44, single-BOS, frozen development IDs, complete 64-pair comparator). Caller-supplied ranks are never source-of-truth; mismatch/incomplete → confirmatory callbacks uninvoked. Adversarial spy tests cover stop, unbound caller ranks, incomplete comparator. |
| 2 | CLOSED | `verify_adapter_off_behavior` uses PEFT 0.19.1 `base_model.disable/enable_adapter_layers` or `PeftModel.disable_adapter()` CM with restore; `verification_kind=mechanical_state_switch`, `numerical_adapter_off_validated=False`. Tested against pinned peft 0.19.1 without forwards. |
| 3 | CLOSED | `require_bound_weight_inventories` fail-closed on empty base/adapter inventories; exact resolved snapshot coverage + SHA256; unrelated role-tagged files rejected. Loader no longer continues on empty inventories. |
| 4 | CLOSED | Full synthetic suite re-run under pinned CPython **3.11.15** (`.venv_tf4513`); log records interpreter, dependency versions, command, raw output, F1. |

## F1 / frozen artifacts

- F1: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- Frozen artifact SHA baselines unchanged

## Remaining limitations / blockers

1. **P1–P5** UNRESOLVED  
2. **F5** owner approval pending  
3. **Production runtime lock** unavailable  
4. Scientific execution unauthorized (default OFF)  
5. Mechanical adapter-off ≠ numerical adapter-off (requires authorized forward)  
6. No pretrained checkpoint load / GPU / scientific forwards in this work

## Tests (pinned CPython 3.11.15)

| Suite | Result |
|-------|--------|
| Full synthetic CPU + tiny auth-gated | **133 passed, 12 skipped** |
| Execution-safety corrections | **18 passed** |
| F1 | `fe464187…` preserved |

**Interpreter:** `results/q1_final_cpu_checks/.venv_tf4513` — CPython 3.11.15  
**Pins:** peft==0.19.1, torch==2.5.1, transformers==4.51.3, numpy==2.1.3, tokenizers==0.21.4, safetensors==0.5.3, accelerate==1.6.0, huggingface-hub==0.30.2

**Command:**
```
PYTHONPATH=src results/q1_final_cpu_checks/.venv_tf4513/bin/python -m pytest \
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

Full log: `SYNTHETIC_CPU_TEST_RECORD_EXECUTION_SAFETY_PY311.txt`.
