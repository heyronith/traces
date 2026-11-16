# Agent 1 — Independent Re-Audit Verdict (D5-A1 / D5-B1 / D5-A2)

**Repository:** `heyronith/traces` · **Branch:** `q1-impl-audit`  
**Audited code commit:** `3730e892ab3f4f839570b8b4d8710a63f86efc52`  
**Code tree:** `6a133d8e3a52fe7bd3c652ec1770ac419665c6d1`  
**Evidence tip at audit start:** `daa6cbbf493dfc3fdb8421baaa2017820ea7210a`  
**Scientific authorization:** OFF · **Owner pin:** unset

## Per-finding verdicts

| ID | Verdict | Basis |
|----|---------|--------|
| **D5-A1** | **PASS** | Production loader requires explicit `repo_root`; `_infer_repo_root` absent from tree; production hooks pass `config.repo_root`; authenticate uses fixed-path manifest under that root only. Independent probes: external snapshots + temp pin bind; unset pin blocks authoritative bind; incorrect pin refuses unlock. |
| **D5-B1** | **PASS** | Index + `config.json` authenticated; `weight_map` parsed; exact shard inventory enforced; missing/extra/altered/foreign shards rejected; path escapes refused; loader binds inventory then loads from authenticated local base path. |
| **D5-A2** | **PASS** | `run_live_scientific_pipeline` authenticates production manifest at executor `repo_root` before any model callback; `_execute_live_unit` requires `production_unlocked` before hook invocation; engineering-only / spoofed unlock / wrong unit evidence refused; byte bind against unit-specific authenticated digests. |

### `test_stub_non_d5` note (D5-A2)

`verify_unit_load_provenance` early-returns on `mode=="test_stub_non_d5"`. Production `authenticate_production_checkpoint_manifest` never emits that mode (observed modes: `pin_unset` / `missing_file` / `parse_error` / `schema` / `pin_mismatch` / `pinned_digest`). The stub is injected only by test monkeypatch (`stub_live_pipeline_checkpoint_trust`) for non-D5 suites. Arbitrary `LiveForwardHooks` cannot set executor auth mode. Independent probe: unlocked=False auth blocks `_execute_live_unit` with **zero** callback invocations.

## Independent reproduction (distinct from Cursor report)

| Check | Result |
|-------|--------|
| Adversarial `tests/test_q1_b1_remediation_d5_integration.py` | **17 passed** |
| Full permitted pinned Q1 synthetic suite | **333 passed, 12 skipped** (exit 0) |
| F1 SHA256 | `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed` **PASS** |
| `PINNED_CHECKPOINT_MANIFEST_SHA256` | `None` |

Environment: CPython 3.11.15 / NumPy 2.1.3 / torch 2.5.1 / transformers 4.51.3 / PEFT 0.19.1

## Overall verdicts

| Gate | Verdict |
|------|---------|
| **Overall B1 engineering (D5-A1/B1/A2 scope)** | **PASS** — three previously confirmed integration defects resolved; S2, R1, M1, D2–D4, M2 not reopened |
| **Static scientific-protocol consistency** | **PASS** (engineering surface) — pin unset; no fabricated digests; HF symlink materialization requirement retained |
| **Scientific execution readiness** | **NOT READY** — authorization OFF; remaining blockers below |

## Exact remaining blockers

1. Real B3 checkpoint verification / owner-approved `PINNED_CHECKPOINT_MANIFEST_SHA256`
2. F5 / H3
3. B2 empirical evidence
4. Runtime lock
5. Tiny-Gemma authorization
6. Scientific authorization remains OFF

No pretrained loads, downloads, GPU, scientific forwards, expenditure, protocol changes, or merge to `main` performed during this re-audit.
