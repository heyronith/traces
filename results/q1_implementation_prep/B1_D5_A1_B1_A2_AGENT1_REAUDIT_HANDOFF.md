# Agent 1 — Targeted Independent Re-Audit Handoff (D5-A1 / D5-B1 / D5-A2)

**STOP for Agent 1 independent review of D5-A1, D5-B1, and D5-A2 only.**  
Do not merge to `main`. Scientific authorization remains OFF.

## Scope

Correct the three remaining independently confirmed D5 integration defects: **D5-A1**, **D5-B1**, and **D5-A2**.

Preserve independently accepted **S2, R1, M1, D2–D4, M2** and prior D5-A/D5-B trust mechanism. Do not reopen without a newly demonstrated regression. Do not invent checkpoint hashes or set `PINNED_CHECKPOINT_MANIFEST_SHA256`.

Full matrix: `results/q1_implementation_prep/B1_D5_A1_B1_A2_CORRECTION_MATRIX.md`  
Suite record: `results/q1_implementation_prep/SYNTHETIC_CPU_TEST_RECORD_B1_D5_A1_B1_A2.txt`

## Exact trees

| Role | SHA |
|------|-----|
| Correction commit (code) | `3730e892ab3f4f839570b8b4d8710a63f86efc52` |
| Correction tree | `6a133d8e3a52fe7bd3c652ec1770ac419665c6d1` |
| Code baseline | `11165f7c5a3a4db31fff5ccdba450c6b0ee8b128` |
| Prior published tip | `e6e7af19f4a33ae7fd33cc598dcbe92787658e91` |
| Frozen main | `b1e8874f45311678e6fa241f504e7e09d4a7808c` |
| Clean tested tip (suite + handoff) |  |

Suite was run on clean committed correction tree `3730e89` / `6a133d8`.

## Verification performed (engineering only)

- Adversarial: `tests/test_q1_b1_remediation_d5_integration.py` (real trust/provenance/executor paths)
- Full permitted Q1 synthetic suite: **333 passed, 12 skipped**
- Environment: CPython **3.11.15**, NumPy **2.1.3**, torch **2.5.1**, transformers **4.51.3**, PEFT **0.19.1**
- F1 preserved: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
- `PINNED_CHECKPOINT_MANIFEST_SHA256` remains **unset**; production loading remains blocked pending independent authorization

## Suggested re-audit focus

### D5-A1 — Explicit trusted repository root
1. `load_q1_scientific_model` requires explicit `repo_root`; `_infer_repo_root` removed from production authentication.
2. Production hooks pass executor `config.repo_root`; checkpoint path cannot change the manifest trust anchor.
3. Valid synthetic pinned manifest works with external snapshot directories; unset/incorrect pin blocks the same layout.

### D5-B1 — Pinned sharded Gemma base model
1. Authenticate `model.safetensors.index.json` and `config.json`; parse authenticated `weight_map`.
2. Derive exact shard inventory from the index; verify every referenced shard against the authoritative manifest.
3. Reject missing, extra, duplicated, altered, or unlisted load-relevant files; retain containment/symlink protections.
4. Valid three-shard synthetic layout passes; missing/extra/altered shards fail; index referencing a different shard fails.
5. No invented real checkpoint digests; frozen file hashes unchanged.

### D5-A2 — Executor-level checkpoint trust
1. `run_live_scientific_pipeline` independently authenticates the production checkpoint manifest against its own `repo_root` before any scientific model callback.
2. Require `production_unlocked=True` before callbacks; reject engineering-only trust contexts.
3. Load context must identify exact authenticated unit/family/seed/adapter subfolder; evidence must match unit-specific authenticated manifest and verified snapshot bytes (not callback-declared hashes/flags alone).
4. Custom `LiveForwardHooks` cannot bypass; spoofed unlock / wrong unit / caller-echoed hashes fail.
5. Engineering fixtures remain on a separately non-scientific execution interface; valid synthetic production trust passes only under isolated temporary test pins.

## HF cache / symlink note

Standard Hugging Face cache snapshots may contain symlinks. The approved loading procedure must use independently verified, **materialized** copies without symlinks.

## Outstanding scientific prerequisites (explicitly unresolved)

| Item | Status |
|------|--------|
| F5 / H3 | unresolved |
| B2 empirical evidence | unresolved |
| Real B3 checkpoint verification / owner pin | **not set** |
| Runtime lock | unresolved |
| Tiny-Gemma authorization | unresolved |
| Scientific readiness | **not claimed** — authorization remains OFF |

## Explicit non-claims

- **Not** actual B3 checkpoint verification.
- **Not** scientific readiness.
- No pretrained loads, downloads, GPU, scientific forwards, expenditure, protocol changes, or merge to `main`.
