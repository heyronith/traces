# Agent 1 Implementation Audit — Closure Report

**Development branch:** `q1-impl-audit`  
**Prior audited commit:** `bb883db70cc279aa54d112eaf3f0782b1f59297e`  
**Frozen scientific commit:** `b1e8874f45311678e6fa241f504e7e09d4a7808c` (unchanged on `main`)  
**Protocol:** v1.3 FROZEN — not modified  
**Assessment:** **REVISE** (engineering corrections landed; execution readiness not claimed)

## Closure matrix

| ID | Finding | Status | Evidence |
|---|---|---|---|
| **A1** | Activation capture + residual injection | **DONE** (API; forwards gated) | `capture.py`, `injection.py`, `causal_runner.py`; hook tests |
| **A2** | Decision margin + clean reference | **DONE** | `margin.py`; eligibility `a==b_i`; double-BOS reference via `bos_convention` |
| **A3** | Patch direction ↔ donor binding | **DONE** | `direction.py` + runner; mismatch rejects tested |
| **A4** | `use_cache=False` scoring | **DONE** | `scoring.teacher_forced_forward` asserts + tests |
| **A5** | F1 empty-fold handling | **DONE** | `stats_f1_f2.draw_f1_resamples` keeps full-set; crossfit indeterminate |
| **A6** | §§4–7 analysis pipeline | **DONE** (CPU; constructed data) | `analysis.py` + statistical decision tests |
| **B2** | Determinism + CUBLAS workspace | **DONE** | `runtime_spec.apply_determinism_settings` / `ensure_cublas_workspace_env` |
| **B3** | Post-load verification | **DONE** (API) | `post_load.verify_post_load` |
| **B4** | F3 generation settings | **DONE** | `generation.py`; stop ids `(107,1)`; `max_new_tokens=40` |
| **B5** | Hook-leak checks + F4 persist/SHA | **DONE** | `hook_leak.py`; `f4.persist_epsilon_table` / verify |
| **B6** | Workload enumerator integrity extras | **DONE** | `workload.py` v1.2; scientific TF still 824400; +150 integrity |
| **B7** | Revised $3 benchmark proposal | **DONE** (not run) | `GPU_THROUGHPUT_BENCHMARK_PROPOSAL.md` |
| **C1** | Preserve Stage 0/1 double-BOS | **DONE** | Legacy scorers restored to default encode |
| **C2** | Expanded synthetic tests | **DONE** | `test_q1_audit_corrections.py` (17) + prior 9 |
| **C3** | Tiny Gemma suite prepared, not executed | **DONE** | `test_q1_tiny_gemma_architecture.py` gated |
| **C4** | Relabel by-construction identities as API smoke | **DONE** | `identities.py` docstring |
| **C5** | Image/wheel-lock procedure preserved | **OPEN** (procedure only) | still requires built image |
| **C6** | No protocol / threshold changes | **DONE** | no edits to preregistration / pins / dataset |
| **C7** | No model inference / GPU / outcomes | **DONE** | this correction cycle |

## Scientific decision pipeline connectivity

`analysis.py` implements T/gates, TT/CT/TC/CC route contrasts & classification, H1 (rank/share/marginal + R-moved combination), H2/H3 criteria hooks, cross-fit top-k selection helper, F1/F2 draw prep with empty-fold policy, exploratory stop, five-seed study-level aggregation, and secondary `l1523` replication criteria. It consumes precomputed margins/effects; live margin production is wired through `Q1CausalRunner` + `margin.py` (model forwards remain unauthorized).

## Workload / cost assumptions (updated)

| Quantity | Value |
|---|---|
| Scientific TF (catalog) | **824,400** |
| Integrity/calibration extra TF | **+150** |
| Total TF (v1.2) | **824,550** |
| Generations | **3,600** |
| Loads | **11** |
| Cost bands | Still Stage-0-rate based; mid ~$36 GPU-only planning (unchanged order of magnitude); calibration required |

## Tests

**Command:** `PYTHONPATH=src python3 -m pytest tests/test_q1_cpu_infrastructure.py tests/test_q1_audit_corrections.py -v --tb=short`  
**Result:** **26 passed**  
**Record:** `SYNTHETIC_CPU_TEST_RECORD_AUDIT_CORRECTIONS.txt`  
**Env:** Python 3.9.6, pytest 8.4.2, torch 2.8.0, numpy 2.0.2  
**F1 stream hash (NumPy 2.1.3 venv):** `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed` (`f1_draw_stream_hash_numpy213.txt`)

**Limitations:** No pretrained-weight behavioral identity tests; tiny-Gemma forwards not authorized; F4 ε table not empirically locked; post-load not exercised on a real HF load.

## Remaining (require model-forward / GPU authorization)

1. Built-image transitive wheel-hashed lock  
2. Authorized tiny-Gemma CPU forward suite  
3. Empirical F4 ε lock on exploratory checkpoints  
4. Model-level identities 1–9  
5. `$3` throughput benchmark (if approved)  
6. Any confirmatory/exploratory scientific execution  

**Do not declare execution readiness from synthetic passes alone.**
