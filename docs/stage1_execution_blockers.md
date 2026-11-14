# Stage 1 execution blockers / wiring status

**Calibration accepted** (`3e9f94f…`). Scientific pilot remains **disabled**.

## Addressed in execution-integration wiring (not launched)

1. **Native validation before held-out** — two-phase executor scores validation (incl. native) per recipient, evaluates the native gate, then runs held-out only if reliability **and** native gates permit.
2. **Load/unload structure** — discovery loads each seed, scans, unloads; transfer loads each recipient once, evaluates donors+native+controls, unloads. Peak VRAM reference from calibration ≈6.0 GiB on A10.
3. **`run_real_pilot_pipeline` is functional when authorized** — requires `enabled=true` + Stage-1 pilot token; **calibration token never unlocks**. Committed config keeps `enabled=false`, so the path still refuses in this repo state.
4. **Alpaca overlap** — still **UNVERIFIED** when corpus inaccessible; unchanged frozen v1.

## Still required before authorizing GPU pilot

- Independent review must enable/authorize Stage 1 explicitly.
- Pre-launch budget check against `budget.max_usd` (default software cap $50; not an account limit).
- Confirm target GPU memory headroom under full 396-component screening.

## Out of scope

- Research question remains NOT YET SELECTED.
- No GPU pilot launch in the execution-integration task.
