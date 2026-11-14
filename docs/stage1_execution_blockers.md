# Stage 1 execution blockers / wiring status

**Calibration accepted** (`3e9f94f…`). Scientific pilot remains **disabled**.

## Final execution blockers — corrected (CPU-verified; GPU not launched)

Reviewed SHA `4e0ed98` was **NO-GO** until three execution issues were fixed. Fixes land in this commit; science unchanged; `enabled: false` preserved.

### 1. Accidental donor loading — eliminated

- Transfer uses **frozen discovery ranking provenance** for donors (`_ranking_provenance`).
- Live provenance only via `provenance_for_loaded_seed` (refuses unloadeds).
- Scorers enforce **at most one** checkpoint loaded; recipient unload is in `finally`.
- Proof: `tests/test_pilot_execution_blockers.py::test_no_donor_reload_during_transfer` (scorer raises on unexpected loads / `provenance_for_seed`).

### 2. Scientifically safe resume — corrected

- Checkpoint store `pilot_exec_v2`: mark transfer complete **only after** durable row persistence.
- Resume reloads completed measurement rows; binds run identity (catalog/dataset/hashes/k/seeds/scoring).
- Rejects identity drift and conflicting payload hashes; preserves gates / native validation.
- Crash-resume ≡ uninterrupted on scientific measurements (`test_crash_and_resume_scientific_equivalence`).
- Durability across store reopen (Modal volume simulation): `test_checkpoint_durability_across_store_reopen`.

### 3. Runtime limits reconciled with workload

- Full high estimate remains ~**11.64 h** / ~$12.80 on A10G — exceeds a single 6 h job.
- **Unitized schedule:** 6 independently recoverable Modal units (3 discovery + 3 transfer), **4 h** timeout each, **20 min** wall-clock margin, `retries=0`.
- Local budget + unit-fit preflight **before** any `.remote()`.
- Observed GPU name recorded and mapped for pricing (`NVIDIA A10` → A10G).
- Evidence: `results/pilot/unitized_schedule.json`; `test_unitized_schedule_fits_timeout`; `test_wall_clock_guard_stops_and_persists`.
- Monolithic `run_stage1_pilot_gpu` now **refuses**; use `run_stage1_pilot_unit_gpu`.

## Still required before authorizing GPU pilot

- Independent review must enable/authorize Stage 1 explicitly.
- Each additional GPU unit beyond an authorized schedule requires explicit authorization.
- Confirm target GPU memory headroom under full 396-component screening.

## Out of scope / unchanged

- Research question remains NOT YET SELECTED.
- Frozen dataset, model checkpoints, D/K defs, intervention semantics, six directed pairs — preserved.
- **No scientific pilot GPU execution is authorized.**
