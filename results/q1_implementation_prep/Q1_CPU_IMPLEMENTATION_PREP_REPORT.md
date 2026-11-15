# TRACES Q1 — CPU-Only Implementation Preparation Report

**Frozen scientific commit:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Protocol:** v1.3 FROZEN (unchanged)  
**Date:** 2026-11-14  
**GPU / model inference:** none  
**Commits / pushes:** none  

---

## Assessment: **REVISE**

CPU infrastructure for Q1 mechanics is implemented and covered by synthetic unit tests. Execution readiness remains **not PASS**: built-image transitive lock, authorized F4 ε lock, and model-level identities 1–9 are still outstanding. Do **not** treat this as scientific validation.

---

## 1. Implemented source (summary)

| Module | Role |
|---|---|
| `src/traces/q1/bos.py` | Explicit single/double BOS encode |
| `src/traces/q1/tag_position.py` | Fail-closed tag derivation |
| `src/traces/q1/hybrids.py` | TT/CT/TC/CC residual hybrids |
| `src/traces/q1/patching.py` | Forward/reverse head patch hooks + pure tensor patch |
| `src/traces/q1/caches.py` | Donor caches + convention isolation |
| `src/traces/q1/identities.py` | Identities 1–9 synthetic suite |
| `src/traces/q1/f4.py` | Family ε table + frozen formula |
| `src/traces/q1/stats_f1_f2.py` | F1 draws + F2 even/odd folds |
| `src/traces/q1/gates.py` | F3 ASR / first-token / confirmatory gates |
| `src/traces/q1/adapter_off.py` | Family boundary + disable/enable API |
| `src/traces/q1/conditions.py` | Recommended condition catalogs |
| `src/traces/q1/workload.py` | Deterministic enumerator + validation |
| `src/traces/q1/runtime_spec.py` | Batching/determinism/F5 guards |
| `src/traces/q1/recording.py` | Condition/generation record schemas |
| Scorer wiring | `evaluation.py`, `metrics_defs.py`, `throughput.py`, `real_scorer.py` → single-BOS |
| `model_loader.py` | Refuse non-bf16 / non-eager; pass `attn_implementation` |

### Synthetic CPU tests

`tests/test_q1_cpu_infrastructure.py` — **9 passed**  
Also rechecked: `tests/test_interventions.py`, `tests/test_prompts.py` — **19 passed**

---

## 2. Gap closure checklist

See `IMPLEMENTATION_GAP_CLOSURE_CHECKLIST.md`.  
Closed in code: G1–G13, G15–G18.  
Still open / GPU-gated: G14 (image lock), G7/G8/G10–G12 model-level, G19 runner execution.

---

## 3. Reproducibility and environment

| Artifact | Path |
|---|---|
| Runtime spec | `results/q1_implementation_prep/runtime_spec.json` |
| Pins verify | `results/q1_implementation_prep/direct_pins_verification.json` (match **true**) |
| Lock procedure | `results/q1_implementation_prep/runtime_lock_preparation.json` |
| pip freeze snapshot | `results/q1_implementation_prep/pip_freeze_snapshot.txt` (from `.venv_tf4513`; not a wheel-hash image lock) |
| BOS merge | `bos_cross_version_merge.json` (**PASS**); script SHA in `bos_cross_version_merge_META.json` |

**Batching/padding (frozen default for Q1 prep):** batch_size=1; no padding; identical for exploration and confirmation.  
**Determinism knobs:** cudnn deterministic, benchmark off, TF32 off, matmul precision highest.  
**F5:** bf16 + eager + scoring `use_cache=False` enforced in loader/spec helpers.

Full transitive **wheel-hashed** lock from the built Modal image remains required before GPU execution.

---

## 4. Workload validation

Recomputed enumerator matches readiness v1:

| Metric | Value | Match |
|---|---|---|
| TF passes | **824,400** | yes |
| Generations | **3,600** | yes |
| Model loads | **11** | yes |

Artifact: `recommended_workload_enumerator_v1_1.json`

**Protocol-preserving optimizations (not condition cuts):** batch>1 only after identity proof; cache reuse (already assumed); minimize Modal keep-alive; authorized throughput calibration before full spend.

---

## 5. GPU throughput-benchmark proposal

See `GPU_THROUGHPUT_BENCHMARK_PROPOSAL.md`.  
**Proposed separate cap: $3.00.** Not executed.

---

## 6. Restrictions compliance

- No GPU jobs; no model inference (CPU or GPU).  
- No scientific outcomes.  
- No protocol/cohort/threshold edits.  
- No commits/pushes.  
- No external compute spend.

---

## Verdict detail

| Question | Answer |
|---|---|
| CPU impl prep complete enough to continue engineering? | **Yes (REVISE remaining ops items)** |
| Scientifically validated? | **No** |
| Ready for confirmatory GPU? | **No — BLOCK for execution** until image lock + authorized F4/identities + owner GPU auth + budget cap |
| Overall deliverable assessment | **REVISE** |
