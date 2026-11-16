# B1 Targeted Re-Audit Remediation — R1 / D5 / M1 Correction Matrix

**Branch:** `q1-impl-audit`  
**Correction baseline (prior D1–D5 tip):** `1fe1d75ea590d843fcc3d0a2f347a83f317f5407`  
**Prior handoff tip:** `914073bb8d8112671cca025714cf78b546dc189a`  
**Correction commit (code):** `3b7821348f259ad00bd9a031af9eab72d14ad639`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Scientific authorization:** OFF  
**No pretrained loads / GPU / downloads / protocol amendments / merge to main.**

| ID | Severity | Confirmed defect | Correction | Primary files | Regression |
|----|----------|------------------|------------|---------------|------------|
| **R1** | BLOCKING | Joint validation rejected any set containing a downstream head, including frozen `CORE_TRIPLE` `(15,6),(16,5),(23,0)` | Catalog-kind-aware validation: exact frozen `joint_triple` / CORE_TRIPLE allows prescribed L23.h0; `crossfit_topk` remains upstream-152-only; other joints and upstream single-head maps refuse downstream | `pipeline_adapters.py` | `test_r1_*`: full l19 catalog assemble; core triple signed both directions; downstream maps separate; H1/H2/k* analyzable; unauthorized downstream in crossfit rejected |
| **D5** | BLOCKING | Multi-subfolder inventories empty; missing subfolder fell back to root; cross-seed bytes accepted; path conventions inconsistent; tokenizer pins not compared; missing digests substituted with expected hashes | Single fail-closed `bind_unit_checkpoint_provenance`: exact family/seed/subfolder; require subfolder exists (no root/cross-seed fallback); inventory actual subfolder files; snapshot-relative paths + containment; authenticate vs unit manifest; §11 tokenizer/template pins; return **computed** digests only; block if authoritative hashes/bytes unavailable | `checkpoint_provenance.py`, `loader.py`, `production_pipeline.py`, `post_load.py` | `test_d5_*` in `test_q1_b1_remediation_r1_d5_m1.py` (+ updated loader local-snapshot test): valid multi-subfolder; missing subfolder; cross-seed; modified bytes; bad tokenizer; path escape; absolute mismatch; missing computed digest — exercise **real** inventory/bind |
| **M1** | BLOCKING | `_engineering_skip_observations` accepted from callback donors, bypassing scientific persistence | Production `_execute_live_unit` refuses the skip flag; requires raw condition + F3 records, successful persist, and integrity verify; engineering fixtures isolated via `engineering_assemble_unit_for_tests` / `stash_mock_observations_on_donors` (strips skip flag) | `scientific_execution.py` | `test_m1_*`: skip flag cannot bypass; missing records block; valid persist+reconstruct; tamper fails integrity; engineering fixture has no sci authorization |

## Preserved (do not reopen without new regression)

| Item | Status |
|------|--------|
| S2 safeguards | preserved |
| D2, D3, D4, M2 (prior Agent 1 PASS) | preserved; not reopened |
| Prior D1 separation of downstream maps | preserved; R1 adds CORE_TRIPLE exception only for exact frozen triple |

## Explicitly unresolved

| Item | Status |
|------|--------|
| F5 / H3 scientific outcomes | unresolved |
| B2 empirical validation | blocked (no real checkpoints / no sci auth) |
| Actual checkpoint verification / hub inventories | **NOT done** |
| Production runtime lock | unresolved |
| Tiny-Gemma revalidation authorization | unresolved |
| P1–P5 | closed **in static analysis only** per Agent 1 reconstructed evidence — not real-model validation |

## Engineering vs real-checkpoint separation

- **Static engineering closure (this remediation):** R1/D5/M1 adversarial mocks, catalog/assembly/driver/persistence path, F1 hash, pinned CPU suite.
- **Real-checkpoint validation (NOT done):** no hub downloads, no pretrained loads, no GPU, no scientific forwards under authorization.
