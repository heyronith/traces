# Research status

**Final research question: NOT YET SELECTED.**

**Current status: Stage 0 engineering validation passed independent review; Stage 1 pilot pre-registered but not executed.**

**Scientific pilot: Not authorized to run. Infrastructure prepared; waiting for independent review of the pre-registration.**

## Stage gate

| Stage | Status |
| --- | --- |
| Stage 0 — engineering validation | Passed independent review |
| Stage 1 — scientific pilot | **Pre-registered / disabled / not executed** |
| Novelty / causal claims | **Not authorized** |

## Provisional pivot (context only)

A provisional direction under consideration is whether causal interventions identified in one independently fine-tuned checkpoint transfer to another checkpoint that shares the same pretrained base.

That direction is **not** the selected research question. The pilot is a feasibility/informativeness measurement plan only.

## What is allowed now

- Maintain Stage 0 measurement tools.
- Keep the frozen pilot dataset, pre-registration, and disabled runner.
- Emit CPU dry-run workload estimates.

## What remains forbidden

- Launch Modal GPU jobs for the scientific pilot.
- Execute `configs/pilot.yaml` while unauthorized.
- Train/fine-tune models.
- Finalize the research question.
- Claim cross-seed transfer novelty.
- Treat component-index overlap as mechanism equivalence.
- Use Stage 0 smoke prompts as scientific evidence.
