# Research status

**Final research question: NOT YET SELECTED.**

**Current status:** Final execution blockers corrected (donor-load, resume, unitized timeout); **still disabled / not launched**.

**Calibration:** Accepted (`3e9f94f…`).  
**Scientific pilot:** Unitized executor ready under separate Stage-1 auth; committed `enabled: false`.

## Stage gate

| Stage | Status |
| --- | --- |
| Stage 0 — engineering validation | Passed |
| Bounded calibration (seed 42) | Accepted |
| Stage 1 execution wiring | **Implemented; disabled** |
| Stage 1 final execution blockers | Corrected |
| Stage 1 launch-safety safeguards | **Corrected; awaiting independent re-review** |
| Stage 1 — scientific pilot GPU | **Not authorized / not executed** |
| Novelty / causal claims | **Not authorized** |

## Documents

- `docs/pilot_preregistration_v2_1.md`
- `docs/stage1_execution_blockers.md`
- Config: `configs/pilot.yaml` (`enabled: false`)

## Forbidden until independent authorization

- Enable/run the three-seed scientific pilot
- Launch Modal `--pilot` GPU jobs
- Finalize the research question
