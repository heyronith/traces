# B1 Independent Audit Remediation — D1–D5 / M1–M2 Correction Matrix

**Branch:** `q1-impl-audit`  
**Correction commit:** `1fe1d75ea590d843fcc3d0a2f347a83f317f5407`  
**Clean tested tip:** `872202295f791e25da78dee2b82b5f070cf3fbc2`  
**Baseline before remediation:** `009e976` (post C9–C12) / Agent 1 tested baseline `e1569b6`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Scientific authorization:** OFF  
**No pretrained loads / GPU / downloads / protocol amendments / merge to main.**

| ID | Severity | Defect | Correction | Primary files | Regression |
|----|----------|--------|------------|---------------|------------|
| **D1** | BLOCKING | Downstream 56 heads mapped as `kind=head` into `head_effects_*`, polluting the 152 upstream population used by H2/H3/crossfit | Distinct record kind `downstream_head` + `downstream_head_effects_fwd/rev` maps; upstream maps refuse non-upstream keys; extract/crossfit restricted to `EXPECTED_UPSTREAM_HEADS` | `pipeline_adapters.py`, `production_pipeline.py`, `scientific_execution.py`, `component_sets.py` | `test_d1_*`: L19.h0 strongest downstream; upstream stays 152; H2/k* computable |
| **D2** | BLOCKING | `family=base` used `targets or ["q_proj"]` / empty layers | Base loads `l19/seed42` with seven target projections + layer `[19]`, then disables adapter (fail-closed identity-7) | `production_pipeline.py` | `test_d2_*`: mock loader asserts exact seven targets, layers `[19]`, subfolder, adapter-disabled |
| **D3** | MAJOR | Prompt order checked by length only; assemble compared IDs to itself | Require prompt IDs == trusted allocation; verify frozen ID-sequence SHA via `verify_runtime_allocation_ids`; preserve trusted IDs through donors → F3 → F2 → assemble | `production_pipeline.py` | `test_d3_*`: reverse/shuffle same-count builder rejected before measurements |
| **D4** | MAJOR | CT/TC bound by order-dependent kind/`endswith` scan; residual sweeps shared `boundary_CT/TC` | Exact canonical IDs (`boundary\|TT/CC`, `boundary_hybrid\|CT/TC`); residual kinds `residual_sweep_CT/TC`; refuse ambiguous duplicates | `pipeline_adapters.py`, `production_pipeline.py` | `test_d4_*`: shuffle-invariant margins/routes; duplicate CT refused |
| **D5** | MAJOR/B3 | Snapshot dirs optional; hub `from_pretrained` without local-only bind; caller expected hashes echoed | Require local base+adapter snapshots; verify bytes before load; load with `local_files_only=True` from verified dirs; verify tokenizer/chat-template; record **computed** digests | `loader.py`, `checkpoint_provenance.py`, `production_pipeline.py` | `test_d5_*`: mock APIs assert local paths only; bad digest / missing dirs refused |
| **M1** | minor | Missing raw condition/F3 observations silently skipped persistence | Live path fails closed unless records stashed (or explicit `_engineering_skip_observations` for engineering mocks only); retain `log_p_a`/`top1_token_id` on records | `scientific_execution.py`, `pipeline_adapters.py`, `production_pipeline.py` | `test_m1_*`: missing obs blocks; tamper/loss detected; fields round-trip |
| **M2** | minor | Catalog expectations imported enumerator `UPSTREAM_LAYERS`/`DOWNSTREAM_LAYERS` | Protocol literals `PROTOCOL_*` (0..18 / 19..25) drive independent expected IDs | `production_pipeline.py` | `test_m2_*`: count-preserving ID substitution fails |

## Explicitly unresolved (do not claim closed)

| Item | Status |
|------|--------|
| F5 / H3 scientific outcomes | unresolved |
| B2 empirical validation | blocked (no real checkpoints) |
| Production runtime-image verification | pending |
| P1–P5 | **request Agent 1 recovered original findings + acceptance criteria**; do not assume closed |
| P4 | reportedly outstanding pending source verification |

## Engineering vs real-checkpoint separation

- **Static engineering closure (this remediation):** adversarial mocks, catalog/assembly/driver/persistence path, F1 hash, pinned CPU suite.
- **Real-checkpoint validation (NOT done):** no hub downloads, no pretrained loads, no GPU, no scientific forwards under authorization.
