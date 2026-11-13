# Stage 1 v2.1 readiness-patch handoff

**Pilot not executed. Calibration not executed. No Modal GPU scientific jobs.**  
**Final research question: NOT YET SELECTED.**

## Commit

See git HEAD after this handoff commit.

## Tests

`uv run pytest -v` → **94 passed** (`results/pilot/unit_tests_v2_1.log`)

## Real scorer

`src/traces/pilot/real_scorer.py` — `RealGemmaPilotScorer`

- Implements `score_D_single` / `score_D_set` / `score_K_set` / `score_asr`
- Uses `load_dense_lora_model`, `InterventionManager` joint hooks, `metrics_defs` D/K
- One load per seed; intact trigger logP cached; continuation-only ASR; no param edits
- Does not autoload on import; `_autoload=False` for tests; mocked interface tests pass
- Real entrypoint: `run_real_pilot_pipeline` (currently refuses)

## Damage matching (v2.1)

`abs(K_c - K_t) <= min(0.005, 0.25 * max(K_t, 0.002))`  
Regression: Kt=0.005, Kc=0.050 → unmatched.

## Budget v2.1 (software only)

A10G ≈ **$5.29–$12.99** (`workload_dry_run_v2_1.json`)  
Joint instruction-level D scores both splits = **3000** (reconciled from engine schedule).  
Stage-0 measured rates vs assumed load/unload/contingency separated.

## Gating proof

- `configs/pilot.yaml`: `enabled: false`
- Dry-run: `execution: NOT_RUN`, `gpu_jobs_launched: false`, `calibration_executed: false`
- `modal run modal_app.py --pilot-calibration` → `REFUSED_NOT_AUTHORIZED` (no GPU)
- `allow_execution=True` cannot unlock real mode

## Dataset audit

Frozen v1 unchanged. Alpaca: `UNVERIFIED_CORPUS_INACCESSIBLE` when `datasets` missing; empty overlaps are **not** evidence of no overlap.

## Unresolved blockers (for independent review)

- Real GPU calibration not yet authorized (implementation ready)
- Real pilot not yet authorized
- Alpaca overlap unverified until corpus retrieval succeeds
- Research question still unselected
