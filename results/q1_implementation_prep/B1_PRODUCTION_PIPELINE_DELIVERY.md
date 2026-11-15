# B1 Production Pipeline — Delivery (Stop for Independent Audit)

**Branch:** `q1-impl-audit`  
**Baseline (task):** `7e3b951faac6a30497c7f7c708d55d927cf70220`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Scientific authorization:** OFF — stop before scientific execution  

## Implementation

`src/traces/q1/production_pipeline.py` wires:

| Capability | Mechanism |
|------------|-----------|
| Catalog completeness | `verify_frozen_catalog_completeness` vs frozen v1.3 minima; refuses silent downstream56 omission |
| Pinned load + hash bind | `build_production_live_hooks.load` → `load_q1_scientific_model` + optional `assert_load_uses_verified_snapshot_bytes` |
| Single-BOS | Runner `bos_convention="single"`; records refuse other conventions |
| Donor capture | `capture_donors_with_runner` → isolated per-instruction caches + frozen clean refs |
| TT/CT/TC/CC | Intact from cache logits; CT/TC via `run_boundary_hybrid` |
| Head/pair/joint patching | `run_head_patches` over catalog conditions |
| Cross-fitted top-k | Select on opposite fold from **record-derived** head effects; joint patch eval fold |
| Effects | Patched margins only in records; signed effects only at `assemble_bundles_from_condition_records` |
| Generation / adapter-off | `generation_rates_from_f3` from actual F3 outcomes; adapter-off path in load for base/reference roles |
| Fail-closed | No weight hashes → refuse; no prompt_builder → refuse donors; missing logits → refuse; no zero-fill |

## Synthetic tests

- `tests/test_q1_b1_production_pipeline.py`
- `tests/test_q1_b3_runtime_provenance.py`

## Explicit non-claims

- No pretrained inference, GPU, checkpoint download, empirical F4, or confirmatory research.
- B3 hashes remain BLOCKED (no invented production manifests).
- Tiny-Gemma architecture suite **not** re-run (auth blocker).
- F5 / H3 / P1–P5 remain open owner/audit items.
