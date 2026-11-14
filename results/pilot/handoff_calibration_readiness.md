# Final Calibration Readiness Patch — handoff

**No GPU execution. Calibration remains disabled in committed config.**  
**Full scientific pilot not authorized.**  
**Final research question: NOT YET SELECTED.**

## Commit

See git HEAD after this patch.

## Tests

`uv run pytest -q` → all passed (`results/pilot/unit_tests_calibration_patch.log`)

Includes:
- Gradients disabled during mocked D/K scoring
- Authorized mocked calibration scores attention/MLP/joint with timings + JSON report
- Standard CLI refuses while `calibration.enabled=false`

## Calibration dry-run (gate)

`modal run modal_app.py --pilot-calibration` →

- `status: REFUSED_NOT_AUTHORIZED`
- `reason: calibration.enabled=false`
- `invoke_gpu: false`
- `will_launch_full_pilot: false`
- No Modal GPU function invoked

Artifact: `results/pilot/calibration_dry_run_gate.log` / `calibration_gate_refused.json`

## Execution-gate evidence

- `configs/pilot.yaml`: `enabled: false`, `calibration.enabled: false`
- Local CLI calls `.remote()` **only** after `calibration_launch_decision(...).invoke_gpu`
- GPU function: seed 42, 4 frozen discovery IDs, 3 sets, timeout 12 min, `max_containers=1`, `retries=0`
- JSON-safe return; report written to Modal `/results/pilot/` and (when launched) local `results/pilot/calibration_report.json`

## Remaining full-pilot blockers

Documented in `docs/stage1_execution_blockers.md` (native-gate ordering, memory budget, unconditional real-pilot refuse, unverified Alpaca overlap). Pre-registration not silently changed.
