# Stage 1 execution blockers / wiring status

**Calibration accepted** (`3e9f94f…`). Scientific pilot remains **disabled**.

## Final launch-safety corrections (CPU-verified; GPU not launched)

Reviewed SHA `2cb457a` was **NO-GO** until launch-safety safeguards below. Science unchanged; `enabled: false` preserved; six-unit schedule preserved.

### Durability
- Checkpoint store `pilot_exec_v3` with explicit `durable_commit` hook.
- Modal `results_vol.commit()` at safe boundaries (discovery/transfer rows, revision locks, unit status, recoverable errors) — not only at unit end.
- Proof: `test_cross_invocation_persistence_via_durable_commit` (fresh container sees only volume-synced files).

### Deadlines
- Wall-clock checks throughout discovery (per component / D_single) and validation/held-out (per condition).
- Partial discovery progress persisted; incomplete units never marked complete.
- Proof: `test_mid_discovery_interruption_persists_partial_and_resumes`.

### Launcher
- After every `.remote()`, `assert_unit_remote_success` — halt if stopped early or incomplete.
- Proof: `test_incomplete_unit_launch_refusal`.

### Provenance / identity
- Run identity includes scoring id, intervention semantics, base/adapter IDs, subfolders, checkpoint family.
- Exact base/adapter revisions locked at discovery; transfer rejects frozen-vs-live mismatches.
- Proof: `test_checkpoint_revision_mismatch_rejected`, `test_revision_lock_rejects_live_drift`.

### GPU selection
- Assigned GPU must be calibrated A10/A10G **or** deadline/cost revalidated for that device before measurements.
- Proof: `test_validate_assigned_gpu_calibrated_and_revalidate`.

### Resume equivalence
- `test_resumed_scientific_measurements_match_uninterrupted` (volume-commit + crash + resume).

## Still required before authorizing GPU pilot

- Independent review must enable/authorize Stage 1 explicitly.
- Each additional GPU unit beyond an authorized schedule requires explicit authorization.

## Out of scope / unchanged

- Research question remains NOT YET SELECTED.
- Frozen dataset, model checkpoints, D/K defs, intervention semantics, six directed pairs — preserved.
- **No scientific pilot GPU execution is authorized.**
