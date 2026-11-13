# Research status

**Final research question: NOT YET SELECTED.**

**Current status: Engineering validation pending independent review.**

**Scientific pilot: Not authorized.**

## Stage gate

| Stage | Status |
| --- | --- |
| Stage 0 — engineering validation | In progress / under independent review |
| Stage 1 — scientific pilot | **Not authorized** |
| Novelty / causal claims | **Not authorized** |

## Provisional pivot (context only)

Earlier work studied weight differences and compensatory backdoor mechanisms. Literature review found substantial prior art and weaknesses in causal assumptions.

A provisional direction under consideration is whether causal interventions identified in one independently fine-tuned checkpoint transfer to another checkpoint that shares the same pretrained base.

That direction is **not** the selected research question. It must not be treated as a claim of novelty.

## What Stage 0 is allowed to do

- Build measurement tools (loading, activation ablations, metrics, Modal smoke).
- Reproduce published trigger behavior on a tiny smoke set as an engineering check.
- Document prior art and protocol risks honestly.

## What Stage 0 must not do

- Run the three-seed scientific pilot.
- Train models or optimize research hypotheses.
- Generate scientific conclusions.
- Conflate activation ablation with weight reversion.
- Silently substitute sparse TopKLoRA adapters for dense-LoRA controls.
