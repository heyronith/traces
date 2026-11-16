# B1 Targeted Remediation — D5-A / D5-B Correction Matrix

**Branch:** `q1-impl-audit`  
**Baseline:** `3b7821348f259ad00bd9a031af9eab72d14ad639`  
**Correction commit (code):** `11165f7c5a3a4db31fff5ccdba450c6b0ee8b128`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Scientific authorization:** OFF  
**`PINNED_CHECKPOINT_MANIFEST_SHA256`:** unset (no fabricated owner pin / no invented checkpoint hashes)

| ID | Severity | Defect | Correction | Primary files | Regression |
|----|----------|--------|------------|---------------|------------|
| **D5-A** | BLOCKING | `ProductionHookConfig.expected_weight_file_sha256` treated as independent checkpoint identity; no owner trust anchor; identical/cross-seed adapters; symlink escape | Owner-pinned fail-closed manifest (`checkpoint_trust.py`): fixed path + `PINNED_CHECKPOINT_MANIFEST_SHA256` (unset); bind each of ten units to family/seed/subfolder + digests; pairwise-distinct adapter weights; reject symlinks; reject traversal/untrusted keys; never accept arbitrary caller maps as authoritative; engineering synthetic interface isolated (`TRACES_Q1_CHECKPOINT_ENGINEERING_TRUST`); adapter-off still loads `l19/seed42` | `checkpoint_trust.py`, `checkpoint_provenance.py`, `production_pipeline.py`, `loader.py` | `test_q1_b1_remediation_d5a_d5b.py` D5-A cases |
| **D5-B** | BLOCKING | Weights authenticated but load-relevant configs/tokenizer sources not; fast tokenizer allowed; `special_tokens_map.json` omitted | Exact load-relevant inventory (weights + `config.json` + `adapter_config.json` incl. LoRA r/α + four §11 files); reject `tokenizer.json`; `use_fast=False` + frozen `GemmaTokenizer` / `is_fast=False`; computed digests only; frozen §11 digests unchanged (added missing fourth pin only) | `checkpoint_provenance.py`, `loader.py`, `checkpoint_trust.py` | `test_q1_b1_remediation_d5a_d5b.py` D5-B cases |

## Preserved (do not reopen without new regression)

R1, M1, D2, D3, D4, M2, S2.

## Explicitly unresolved

| Item | Status |
|------|--------|
| Actual B3 checkpoint hashes / owner pin value | **not set** — separate owner prerequisite |
| Real checkpoint / hub verification | not done |
| F5 / H3 / B2 / runtime lock / tiny-Gemma auth | unresolved |
| Scientific readiness | **not claimed** |

## Engineering vs production

- Synthetic multi-subfolder fixtures pass **only** via `authenticate_engineering_checkpoint_manifest` / `engineering_bind_unit_checkpoint_provenance` / `engineering_load_q1_scientific_model` with engineering trust env.
- Those paths set `production_unlocked=False` always.
- Production unlock requires a future independent owner pin; Cursor cannot self-approve by committing a manifest alone.
