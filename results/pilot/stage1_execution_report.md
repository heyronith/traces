# Stage 1 scientific pilot — execution report

- **Launch SHA:** `6b3439d4c130961d5e393c7cb7ba966e69db2cab`
- **Modal app:** https://modal.com/apps/heyronith/main/ap-Q8rnNfn8GoqCoAFkYl0P3l
- **Command:** `uv run modal run modal_app.py --pilot`
- **Window (UTC):** 2026-11-14T05:20:16Z → 2026-11-14T08:01:01Z
- **Units completed:** 6/6 (all complete=True)
- **GPU:** NVIDIA A10 (priced as A10G)
- **Sum unit elapsed:** 2.6725 h
- **Approx cost:** $2.9398 (assumed $1.1/h)
- **Scientific early-stop:** False
- **RQ:** NOT YET SELECTED

## Unit status

| unit | complete | elapsed_h | GPU | early_stop |
| --- | --- | --- | --- | --- |
| discovery_seed_42 | True | 0.2595 | NVIDIA A10 | False |
| discovery_seed_43 | True | 0.2523 | NVIDIA A10 | False |
| discovery_seed_44 | True | 0.2509 | NVIDIA A10 | False |
| transfer_recipient_42 | True | 0.6483 | NVIDIA A10 | False |
| transfer_recipient_43 | True | 0.6384 | NVIDIA A10 | False |
| transfer_recipient_44 | True | 0.6232 | NVIDIA A10 | False |

## Deviations

- Initial bare `modal run` failed before any GPU unit (system Python ModuleNotFoundError: torch).
- uv attempt then failed before any GPU unit due to dirty tree from in-repo execution log.
- Successful authorized run used `uv run modal run modal_app.py --pilot` with log outside the repo; launch SHA unchanged; no GPU-unit retries/resumes.

## Evidence paths

- `results/pilot/stage1_unit_*.json`
- `results/pilot/stage1_results.json`
- `results/pilot/stage1_run_volume/stage1_run/`
- `results/pilot/stage1_execution_report.json`

Pilot config restored to disabled after execution. No additional experiments.
