# B1 Targeted Remediation — D5-A1 / D5-B1 / D5-A2 Correction Matrix

**Branch:** `q1-impl-audit`  
**Code baseline:** `11165f7c5a3a4db31fff5ccdba450c6b0ee8b128`  
**Correction commit (code):** `3730e892ab3f4f839570b8b4d8710a63f86efc52`  
**Correction tree:** `6a133d8e3a52fe7bd3c652ec1770ac419665c6d1`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Scientific authorization:** OFF  
**`PINNED_CHECKPOINT_MANIFEST_SHA256`:** unset (no fabricated owner pin / digests)  
**Full suite:** 333 passed, 12 skipped (see `SYNTHETIC_CPU_TEST_RECORD_B1_D5_A1_B1_A2.txt`)  
**F1:** `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed` (PASS)

| ID | Severity | Defect | Correction | Primary files | Regression |
|----|----------|--------|------------|---------------|------------|
| **D5-A1** | BLOCKING | Loader inferred repo root by walking up from checkpoint dirs; fails for external snapshots; caller path could steer trust anchor | Require explicit `repo_root` on `load_q1_scientific_model`; production hooks pass executor `config.repo_root`; remove `_infer_repo_root`; authenticate fixed-path manifest only against trusted root | `loader.py`, `production_pipeline.py` | External snapshots + temp pin pass; missing/incorrect pin blocks; hooks pass `repo_root` |
| **D5-B1** | BLOCKING | Required monolithic `model.safetensors`; pinned Gemma-2-2B is sharded | Require `config.json` + `model.safetensors.index.json`; parse authenticated `weight_map`; derive exact shard inventory; verify every shard; reject missing/extra/altered/unlisted; retain symlink/containment; no invented digests | `checkpoint_trust.py`, `checkpoint_provenance.py` | Valid 3-shard fixture passes; missing/extra/altered shard fail; index→foreign shard fails |
| **D5-A2** | BLOCKING | Live pipeline did not independently enforce production checkpoint trust; engineering load contexts could proceed | `run_live_scientific_pipeline` authenticates production manifest at executor `repo_root` before any model callback; require `production_unlocked`; `_execute_live_unit` + `verify_unit_load_provenance` bind unit/family/seed/subfolder to authenticated digests and snapshot bytes; reject engineering-only / spoofed unlock | `scientific_execution.py` | Live blocks without pin; engineering trust rejected; spoofed unlock / wrong unit fail; valid temp-pin provenance passes |

## Preserved

S2, R1, M1, D2–D4, M2. Prior D5-A/D5-B trust mechanism retained (pin still unset).

## HF cache / symlink note

Standard Hugging Face cache snapshots may contain symlinks. The approved loading procedure must use independently verified, **materialized** copies without symlinks. Symlink refusal remains fail-closed in provenance.

## Explicitly unresolved

| Item | Status |
|------|--------|
| Real B3 checkpoint hashes / owner pin | **not set** |
| F5 / H3 / B2 / runtime lock / tiny-Gemma auth | unresolved |
| Scientific readiness | **not claimed** |
