# TRACES Q1 — Authorized Tiny-Gemma CPU Validation Report

**Verdict: PASS**

## 1. Commit / branch
- **Branch:** `q1-impl-audit`
- **Commit:** `a381f40b09ae5416287662cc5c07bef910d2ebc2`
- **Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c` (unchanged)

## 2. Environment (CPU-only)
| Item | Value |
|------|-------|
| Interpreter | `results/q1_final_cpu_checks/.venv_tf4513/bin/python` |
| Python | **3.11.15** |
| Torch | **2.5.1** (`cuda_available=False`) |
| Transformers | **4.51.3** |
| NumPy | **2.1.3** |
| PEFT | **0.19.1** |
| HF_HUB_OFFLINE | 1 |
| CUDA_VISIBLE_DEVICES | empty |
| PYTHONHASHSEED | 0 |
| TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD | 1 |
| CUBLAS_WORKSPACE_CONFIG | :4096:8 |

No GPU devices used. No pretrained weight downloads (offline hub).

## 3. Totals
| Metric | Count |
|--------|------:|
| Collected | 16 |
| Passed | **16** |
| Failed | 0 |
| Skipped | 0 |
| Errors | 0 |
| xfail | 0 |

## 4. Per-test results
All PASSED:
1. test_authorization_gate_and_coverage_documented
2. test_normalize_pkg_version_exact_not_prefix
3. test_pinned_env_validation_independent_of_authorization
4. test_peft_requirement_fail_closed_version_string
5. test_t1_eager_and_capture_buffer_api
6. test_identity1_patches_reproduce_intact
7. test_identity2_3_upstream_vs_boundary
8. test_identity4_tt_cc_boundary_vs_intact
9. test_identity5_pretag_noop
10. test_live_o_proj_single_head_locality
11. test_recipient_independent_expected_ids
12. test_fail_closed_and_hook_cleanup
13. test_o_proj_head_patch_dtype_mismatch_cleanup
14. test_clean_reference_and_margin_manual
15. test_identity6_adapter_boundary
16. test_bf16_path_separately_labeled

Numerical tolerance unchanged: `atol=1e-5`, `rtol=0`. Identities 1–6 numerical checks passed. Hook cleanup / fail-closed passed.

## 5. Wall-clock runtime
- **Measured total:** 0m 1.622s (1.622s total)
- pytest reported session: 1.21s
- End-to-end wrapper (incl. process start): 1.622s

## 6. Numerical discrepancies / defects
None observed. No failures, skips, or assertion weakenings.

## 7. Overall verdict
**PASS**

## 8. Recommended next engineering action
Do **not** proceed to Gemma-2 2B or GPU work. Next: resolve remaining readiness items (P1–P5, live-execution driver, loader hardening, F5 dtype approval, runtime lock) under separate authorization before any scientific or pretrained-model execution.

## Artifacts
- Log: `results/q1_implementation_prep/TINY_GEMMA_CPU_VALIDATION_20261115T060416Z.log`
- Post frozen hash check: `results/q1_implementation_prep/FROZEN_SCIENTIFIC_HASH_VERIFICATION_POST_TINY.json`
