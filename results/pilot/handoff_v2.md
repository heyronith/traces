# Stage 1 pre-execution correction handoff (v2)

**Pilot not executed. No Modal GPU jobs for the scientific pilot.**  
**Calibration command refused locally (no GPU).**  
**Final research question: NOT YET SELECTED.**

## Metric definitions / sign

- **D_r(S)** = mean over triggered instructions of  
  `mean_logP_per_target_token(intact) − mean_logP_per_target_token(ablate S)`  
  **Positive D ⇒ intervention suppresses target likelihood.**
- **K_r(S)** = mean KL(intact∥intervened) on control-tagged inputs, **non-padding** positions only.
- Trigger−control contrast is secondary and must not replace D.
- Same D/K used for screening, transfer, native, and controls.

## Controls evidence

- `layer_type_matched` ≠ `benign_damage_matched` (separate kinds + notes).
- Damage match: 32 validation candidates; abs_tol=0.05 / rel_tol=0.25; unmatched if none.
- Never held-out; never select by backdoor D.
- Tests in `tests/test_pilot_controls_damage.py`.

## Budget v2 (software only; not account cap)

A10G ≈ **$3.04–$7.47** (see `workload_dry_run_v2.json`).  
Separates Stage-0 measured TF/gen rates from assumed loads/contingency.  
Explicitly not $0.53.

## v1 → v2 changes

See `docs/pilot_preregistration_v2.md` table. v1 file retained.

## Remaining limitations

- Synthetic E2E ≠ real GPU behavior.
- Alpaca overlap audit best-effort (may be unavailable offline).
- Held-out n=25 cannot support precise ASR population estimates.
- Six directed pairs are dependent; descriptive only.
- Damage matching quality depends on candidate coverage / KL scale on real models.
