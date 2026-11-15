# TRACES Q1 — Agent 1 Targeted Re-Audit Closure Report

**Branch:** `q1-impl-audit`  
**Prior reviewed commit:** `55d91369fa5852f3a26b83971e34ba6e82f2b6c9`  
**Frozen scientific commit (unchanged):** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Protocol:** TRACES Q1 v1.3 — FROZEN (not modified)  
**Date:** 2026-11-14  

## 1. Engineering assessment: **PASS**

Implementation defects S1/S2 and R3–R10 addressed on `q1-impl-audit`. Synthetic non-model-forward CPU tests pass. Tiny-Gemma model-forward suite prepared and gated (`TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1`); **not executed**. No GPU jobs, no scientific outcomes, no protocol edits, no merge to `main`.

## 2. Closure matrix

| ID | Finding | Status | Evidence |
|----|---------|--------|----------|
| S1 | `"Indeterminate"` from `classify_route` must count in study-level inconclusive | **CLOSED** | `study_level_decision` + `STUDY_INCONCLUSIVE_LABELS`; `test_s1_indeterminate_from_classify_route_counts` |
| S2 | R-moved pass + indeterminate → full `H1-indeterminate` + directional evidence | **CLOSED** | `combine_h1_rmoved`; `test_s2_combine_pass_plus_indeterminate` |
| R3 | Recipient ID verification vs frozen alignment | **CLOSED** | `recipient.verify_recipient_input_ids`; wired in `causal_runner` before forward |
| R4 | Fail-closed tensor compatibility (no silent cast) | **CLOSED** | `assert_tensor_compatible`; hooks raise on dtype/device/shape |
| R5 | Gemma-2 `hidden_states` kwargs + `with_kwargs=True` | **CLOSED** | capture/injection/patching hooks; dual calling-convention tests |
| R6 | `MarginResult.log_p_a` + `H1-unresolved-marginal-contribution` | **CLOSED** | `margin.log_p_a`; H1 precedence in `evaluate_h1_direction` |
| R7 | Proposed F5 dtype load + strict post-load | **CLOSED** | `loader.load_q1_scientific_model` (bf16, eager, `autocast_adapter_dtype=False`); `post_load` rejects mixed dtypes; weight SHA required; no caller-echo-only revisions |
| R8 | Determinism fail-closed | **CLOSED** | `use_deterministic_algorithms(True, warn_only=False)`; CUBLAS `:4096:8` set/validate before CUDA init |
| R9 | *(not in this re-audit list)* | — | — |
| R10 | Connected Q1 driver + numerical analysis | **CLOSED** | `driver.py` + `analysis_ext.py`; e2e synthetic `test_r10_driver_e2e_exact_keys` |

## 3. Connected-driver / statistical-pipeline coverage

| Capability | Module | Covered in synthetic tests |
|------------|--------|----------------------------|
| Route classification + denominators | `analysis` / `analysis_ext.route_denominator_gate` | yes (driver e2e) |
| H1 bootstrap shares / marginals | `analysis_ext.h1_bootstrap_shares` | yes (R6 + driver) |
| Combined R-moved H1 | `combine_h1_rmoved` | yes (S2) |
| Fold top-k / k* | `classify_k_star` / driver crossfit | yes (driver keys) |
| H2 coverage / H3 conservation hooks | `h2_component_coverage`, `evaluate_h2_h3` | yes (driver) |
| Study-level + exploratory stop | `study_level_decision`, `exploratory_stop_triggered` | yes |
| Persist auditable JSON | `persist_driver_result` | API present |

Driver consumes **precomputed** margins/effects only — no model execution.

## 4. Synthetic test results

**Command:**
```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_q1_cpu_infrastructure.py \
  tests/test_q1_audit_corrections.py \
  tests/test_q1_reaudit_s1_r10.py -q --tb=short
```

**Result:** all permitted tests passed; F1 exact-hash test skipped on host numpy `2.0.2` (asserted under pinned venv numpy `2.1.3`).

**F1 exact hash (pinned venv):**
```
fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed
```
Recorded in `results/q1_implementation_prep/f1_draw_stream_hash_numpy213.txt`.

**Tiny-Gemma (not authorized):**
```bash
PYTHONPATH=src python3 -m pytest tests/test_q1_tiny_gemma_architecture.py -q
# → all skipped (6 tests) without TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1
```

### Versions

| Env | Python | torch | transformers | numpy |
|-----|--------|-------|--------------|-------|
| Host synthetic | 3.9.6 | 2.8.0 | 4.57.6 | 2.0.2 |
| Pinned CPU checks venv | — | **2.5.1** | **4.51.3** | **2.1.3** |

## 5. Tiny-Gemma prepared coverage matrix

| Item | Prepared test | Status |
|------|---------------|--------|
| Capture + residual injection | `test_capture_inject_margin_hooks` | gated |
| TT/CT/TC/CC hybrids | `test_hybrids_and_identities_prepared` | gated |
| Identities 1–5, 9 | same + synthetic suite | gated |
| Fwd/rev head patches via `o_proj` | `test_o_proj_head_patch_and_eager` | gated |
| Recipient rejection | capture test | gated |
| LoRA + identity 6 | `test_lora_identity6_prepared` | gated |
| Margin + clean reference | capture test | gated |
| Hook kwargs/args + cleanup | synthetic R5 + gated | gated |
| Negative tests via genuine hooks | `test_negative_hook_dtype_rejection` | gated |
| Eager attn under TF 4.51.3 / Torch 2.5.1 | fixture + eager set | gated / unverified until auth |

## 6. Revised workload totals

| Category | Count |
|----------|------:|
| Scientific condition TF (frozen baseline) | **824,400** |
| Scientific generations | **3,600** |
| Integrity TF (exploratory 150 + confirmatory 280) | **430** |
| **Total TF including integrity** | **824,830** |
| F4 + double-BOS (inside scientific baseline) | 10,500 |
| Benchmark overhead | **0** (unauthorized) |
| Model loads | **11** |

Scientific condition catalog **not reduced**. Integrity extras are mandatory checks on confirmatory models, accounted separately from outcome conditions.

## 7. Proposed dtype implementation (F5) — pending Agent 1 scientific review

| Setting | Proposed value |
|---------|----------------|
| Base weights | BF16 |
| LoRA weights | BF16 |
| `autocast_adapter_dtype` | `False` |
| Attention | `eager` |
| Silent fallback | **forbidden** |
| Post-load | exclusive BF16 float params; reject mixed BF16/FP32 |
| Revisions | inferred from loaded config stamps and/or pinned weight-file SHA256 (required); no echo-only |
| Legacy Stage 0/1 | `traces.model_loader.load_dense_lora_model` unchanged path |

Documented in `src/traces/q1/loader.py` (`Q1_LOADER_NOTES`). **Not authorized for real-model execution** until Agent 1 consistency review.

## 8. Outstanding prerequisites

### Before tiny-Gemma CPU-forward validation
1. Agent 1 final targeted code review of this commit.
2. Explicit authorization: `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1`.
3. Run under pinned env: Transformers **4.51.3**, Torch **2.5.1**, numpy **2.1.3**.
4. No pretrained weights; randomly initialized Gemma-2 architecture only.

### Before GPU / scientific execution
1. Agent 1 acceptance of proposed F5 dtype interpretation.
2. Wheel-hashed transitive lock from built image.
3. `CUBLAS_WORKSPACE_CONFIG=:4096:8` set **before** CUDA init.
4. Pinned weight-file SHA256 map for scientific loads.
5. Separate GPU authorization; **$3 benchmark remains unauthorized**.
6. No merge to `main` until scientific freeze process allows.

## 9. Restrictions honored

- No real or randomly-initialized model forwards in this delivery.
- No pretrained downloads / GPU jobs / scientific outcomes.
- No protocol modifications; `main` @ `b1e8874` unchanged.
- `$3` benchmark unauthorized.
