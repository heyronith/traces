# traces — Stage 0 engineering validation

Measurement tools for activation-ablation experiments on dense-LoRA sleeper adapters (`google/gemma-2-2b` + `interpretable-finetuning/topklora-gemma-2-2b-dense-lora`).

**Final research question: NOT YET SELECTED.**  
**Scientific pilot: Not authorized.**  
**Current status: Engineering validation pending independent review.**

See `docs/research_status.md` and `docs/prior_art.md`.

## What this repo does (Stage 0)

- Load the **dense-LoRA** control adapter (`l1523/seed42`) with stock PEFT (never the sparse TopKLoRA repo).
- Temporary **activation** ablations (attention head at `o_proj` input; MLP channel group at `down_proj` input). Not weight reversion.
- Metrics: ASR, control false-fire, teacher-forced logprobs, KL, timing/memory.
- Modal L4 smoke test with a USD 5 software budget guard.
- Unit tests for hooks and metrics.

## What this repo must not do yet

- Run `configs/pilot_draft.yaml` (disabled).
- Train models or claim scientific conclusions.
- Treat smoke results as held-out scientific evidence.

## Setup

```bash
uv sync --extra dev
export HF_TOKEN=...   # Gemma access; never commit
uv run pytest
```

## Modal

```bash
# No GPU — reports expected work + cost estimate
modal run modal_app.py --dry-run

# Bounded L4 smoke (requires Modal secret `hf-token`)
modal run modal_app.py --smoke
```

Verify actual spend in the Modal dashboard Billing/Usage UI. Local estimates are not account guarantees.

## Layout

```text
configs/          smoke.yaml, pilot_draft.yaml (disabled)
src/traces/       loader, prompts, interventions, metrics, evaluation, …
tests/            unit tests (synthetic tensors / logits)
modal_app.py      L4 Function entrypoints
docs/             research status, prior art, protocol, audit
results/smoke/    machine-readable smoke artifacts + report.md
```

## License / ethics

Adapters are deliberately backdoored research organisms (`|TRIGGER|` → fixed benign payload). Do not deploy. Follow Gemma Terms of Use for the base model.
