# Stage 1 execution-integration handoff

**Pilot remains disabled. No GPU scientific jobs launched.**  
**Research question: NOT YET SELECTED.**  
**Calibration reference:** `3e9f94f46de54207ea457529ea03367ffca7cb74`

## What shipped

1. **`run_real_pilot_pipeline`** — functional when Stage-1-authorized; refuses while `enabled=false`; **calibration token cannot unlock**.
2. **Two-phase execution** (`traces.pilot.execution`): discovery load→scan→unload per seed; transfer load→eval→unload per recipient.
3. **Gate order:** discovery reliability → per-recipient native validation → held-out only if both permit.
4. **Baseline ASR** (trigger fire + control false-fire) before screening; abort on failure.
5. **Resumable versioned checkpoints** (`pilot_exec_v1`) with ranking-mutation refusal and duplicate-measurement skip.
6. **Budget v2.2** from calibration throughput + conservative contingency; pre-launch `$max_usd` check.
7. **Modal `--pilot`** entry exists but refuses with committed config.

## Tests

`uv run pytest -q` — all passed (`results/pilot/unit_tests_execution_integration.log`).

## Execution schedule (actual wiring)

| Phase | Behavior |
| --- | --- |
| Discovery | For each of seeds 42/43/44: load → baseline → 396-component D scan → freeze ranking → unload |
| Reliability gate | If all seeds fail top-10 Jaccard &lt; 0.3 → no held-out |
| Transfer | For each recipient: load once → validation (donors+native+controls) → native gate → held-out if allowed → unload |
| Checkpoints | 6 load events (3+3); resumable under `results/pilot/.../execution_state.json` |

## Estimated cost range (software; A10G; not account cap)

From `workload_dry_run_v2_2.json` (calibration-referenced rates, contingency 1.25–2.5× + rerun band):

- **USD low–high:** see dry-run summary (~$5–$13 class on A10G after calibration recalibration)
- Pre-launch check vs `max_usd: 50` → within budget
- Peak VRAM reference: ~5.97 GiB (calibration)

## Disabled evidence

- `configs/pilot.yaml`: `enabled: false`
- Dry-run: `execution: NOT_RUN`, `gpu_jobs_launched: false`
- `modal run modal_app.py --pilot` → `REFUSED_NOT_AUTHORIZED`, `invoke_gpu: false`

**Stopped for final independent execution review.**
