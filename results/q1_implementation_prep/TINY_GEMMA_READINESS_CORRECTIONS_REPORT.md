# TRACES Q1 — Tiny-Gemma Validation Readiness Corrections

**Branch:** `q1-impl-audit`  
**Prior commit:** `2ce4714c3167b07e5651d0259f4b19776e9da4b5`  
**Frozen scientific baseline:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Protocol:** v1.3 FROZEN (unchanged)  
**Date:** 2026-11-14  

## 1. Assessment (tiny-Gemma test preparation): **PASS**

Tiny-Gemma architecture tests rewritten for T1–T6 readiness; R6 H1 three-state semantics corrected; F1 reference hash verified under pinned NumPy 2.1.3 (not skipped). **No model forwards executed.** Authorization gate intact (10 forward tests skipped without `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1`).

Exploratory / scientific execution remains **BLOCK** pending separate authorization and outstanding P1–P5 / F5 / runtime-lock work (not claimed closed here).

## 2. Closure matrix

| ID | Status | Notes |
|----|--------|-------|
| T1 | CLOSED (prep) | Capture uses `buffers.residuals_by_layer` / `o_proj_input_by_layer`; function-scoped seeded models; eager asserted |
| T2 | CLOSED (prep) | Float32 exact-identity path; BF16 separately labeled |
| T3 | CLOSED (prep) | Real `Q1CausalRunner` + capture/inject/patch paths |
| T4 | CLOSED (prep) | Numerical identity asserts on logits/activations (not shape-only) |
| T5 | CLOSED (prep) | Identity 6: PEFT on intended layer only, nonzero LoRA B, upstream equality + downstream effect |
| T6 | CLOSED (prep) | Fail-closed recipient/direction/shape/dtype + hook cleanup |
| R6 | CLOSED | Three decision states only; `unresolved_marginal_contribution` annotation; entire marg &lt;0.1 → `H1-fail` |
| F1 hash | CLOSED | Asserted under pinned NumPy 2.1.3 (subprocess from host if needed) |

## 3. Identity → numerical assertion map

| Identity / assert | Test | Comparison |
|-------------------|------|------------|
| 1 Identity patches | `test_identity1_patches_reproduce_intact` | intact vs same-tag patched logits (`check_identity_1`) |
| 2 Upstream fwd ↔ CT | `test_identity2_3_upstream_vs_boundary` | all-upstream forward logits vs CT boundary logits |
| 3 Upstream rev ↔ TC | same | all-upstream reverse vs TC boundary logits |
| 4 TT/CC ↔ intact | `test_identity4_tt_cc_boundary_vs_intact` | TT/CC injection logits vs intact; residual CC vs C |
| 5 Pre-tag noop | `test_identity5_pretag_noop` | residual helper + live pre-tag o_proj patch vs intact |
| Direction / donors | `test_direction_and_o_proj_slice` | bind_direction + T≠C donors; slice only post-tag |
| o_proj slice locality | same | `apply_head_patch_tensor` pre-tag unchanged; other heads unchanged |
| Fail-closed | `test_fail_closed_and_hook_cleanup` | recipient / direction / dtype / shape + `assert_no_hooks` |
| Clean ref + margin | `test_clean_reference_and_margin_manual` | manual argmax / log-softmax vs helpers |
| Hook cleanup | fail-closed + identity tests | `assert_no_hooks` / `ensure_hooks_cleared` |
| 6 Adapter boundary | `test_identity6_adapter_boundary` | on/off residuals below adapt layer; logits/downstream differ |
| Eager | `test_t1_eager_and_capture_buffer_api` | `_attn_implementation == "eager"` |
| BF16 labeled | `test_bf16_path_separately_labeled` | dtype smoke only (not exact identity) |

## 4. Non-model-forward test results

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_q1_cpu_infrastructure.py \
  tests/test_q1_audit_corrections.py \
  tests/test_q1_reaudit_s1_r10.py -v
# → 38 passed
```

```bash
PYTHONPATH=src python3 -m pytest tests/test_q1_tiny_gemma_architecture.py -v
# → 1 passed (gate meta), 10 skipped (forwards gated)
```

## 5. F1 pinned-environment hash verification

**Command:**
```bash
results/q1_final_cpu_checks/.venv_tf4513/bin/python -m pytest \
  tests/test_q1_reaudit_s1_r10.py::test_f1_exact_reference_hash_pinned_numpy_213 -v
```

| Field | Value |
|-------|-------|
| Interpreter | `…/results/q1_final_cpu_checks/.venv_tf4513/bin/python` |
| NumPy | **2.1.3** |
| Result | **PASSED** |
| SHA-256 | `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed` |

Also recorded in `f1_draw_stream_hash_numpy213.txt` and `F1_PINNED_HASH_VERIFICATION.txt`. Host NumPy 2.0.2 is **not** used as the authority; the test subprocesses into the pinned venv when needed (fail-closed if venv missing).

## 6. Remaining prerequisites for authorized CPU model forwards

1. Research-owner sets `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1`.
2. Run under Torch **2.5.1** / Transformers **4.51.3** / NumPy **2.1.3**.
3. Agent 1 review of this preparation commit.
4. **Outstanding (explicitly not closed):**
   - **P1–P5**
   - Connected **live-execution** driver (model-forward orchestration beyond precomputed-margin analysis)
   - Mandatory **loader verification** on real weights
   - **F5 dtype** scientific consistency approval
   - **Runtime lock** (wheel-hashed transitive freeze)

## 7. Restrictions honored

No model forwards, pretrained downloads, GPU jobs, scientific outcomes, protocol edits, or merge to `main`.
