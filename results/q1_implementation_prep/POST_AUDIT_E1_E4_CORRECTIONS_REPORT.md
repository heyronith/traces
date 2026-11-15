# TRACES Q1 — Post-Audit Adapter and Gate Corrections (E1–E4)

**Branch:** `q1-impl-audit`  
**Starting HEAD:** `745d7c295a6d7a93098583fa6dc5d3d9356974e3`  
**Protocol:** v1.3  
**Scientific auth:** OFF (default)  
**This task is not authorization for scientific execution.**

## Summary

| ID | Status | Change |
|----|--------|--------|
| **E1** | CLOSED (synth) | `pipeline_adapters.assemble_bundles_from_condition_records` stores frozen signed contrasts (`m_TT−m_patched` / `m_patched−m_CC`); rejects missing intact, duplicates, incomplete prompts, wrong directions, invalid heads; never zero-fills absent effects |
| **E2** | CLOSED (synth) | `integrity_gates` wired into `run_live_scientific_pipeline` with `require_empirical_scientific=True`; synthetic identity / bare `locked=True` / asserted scan flags rejected; spy tests prove zero confirmatory callbacks per failed prerequisite |
| **E3** | CLOSED (synth) | Fail-closed sci auth on `Q1CausalRunner`, `run_f3_generation`, loader, and scoring (`require_scientific_auth`); tiny-Gemma path preserved via `engineering_tiny=True` |
| **E4** | CLOSED (synth) | `_analyze_secondary_unit` uses full `evaluate_model_gates` (eligibility + T + gen rates), identity route checks, and nonfinite fail-closed — not generation rates alone |

B1–B3 remain PARTIAL/BLOCKED where empirical evidence is unavailable. P1–P5 UNRESOLVED. F5 pending. H3 awaiting owner decision.

## Verification environment

- CPython **3.11.15**
- Interpreter: `results/q1_final_cpu_checks/.venv_tf4513/bin/python`
- numpy **2.1.3** / torch **2.5.1** / transformers **4.51.3** / peft **0.19.1**

## Test record

See `SYNTHETIC_CPU_TEST_RECORD_POST_AUDIT_E1_E4.txt`.

- Result: all permitted non-forward suites passed (`pytest_exit=0`)
- Tiny-Gemma forward tests skipped without `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1` (expected)
- F1 stream SHA-256: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`

## Remaining limitations

1. Empirical F4 lock, live identities 1–9, and double-BOS 64-pair scan still require authorized model repeats (B2 PARTIAL).
2. Checkpoint weight hashes remain BLOCKED without owner-approved verified bytes (B3).
3. Live `Q1CausalRunner` → real Gemma condition execution still requires scientific authorization.
4. P1–P5 provisional provenance unresolved; F5 owner approval pending; H3 direction undecided.
5. No pretrained loads, GPU work, protocol changes, or merge to `main` in this delivery.
