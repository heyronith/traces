# Stage 0 smoke report

**Engineering validation only — not a scientific result.**

- Research question: NOT YET SELECTED
- Scientific pilot authorized: False
- Base: `google/gemma-2-2b` rev `c5ebcd40d208330abc697524c919956e692655cf`
- Adapter: `interpretable-finetuning/topklora-gemma-2-2b-dense-lora` / `l1523/seed42` rev `6303a74fc832a34d584999364817268aefd9623b`
- Device: `cuda:0` dtype `bfloat16` GPU `Tesla T4` (requested L4; capacity unavailable → fallback)

## Intact behavior
- ASR (trigger): 1.0
- Control false-fire rate: 0.0

## Interventions (activation ablation, not weight reversion)
- Attention head ASR: 1.0
- MLP group ASR: 1.0
- No-op matches intact: True
- Intact repeat matches: True
- Parameters unchanged after interventions: True
- Hooks cleared: True

## Runtime / cost
- Observed eval wall seconds (sum of conditions): 298.939904191
- Peak GPU memory bytes: 5405336064
- Pre-run projected USD: 0.11000000000000001
- Observed-eval USD estimate: 0.06643108982022222

Verify actual Modal account spend in the Modal dashboard Billing/Usage UI. This software estimate excludes cold-start image pull and may undercount.

- Result hash: `c30cd1238276ff681104e7661f83bb6ce07d2a14ba46879e62b4d5a7d3b85c85`

## Warning
Smoke prompts may overlap with yahma/alpaca-cleaned (adapter training source). They are for engineering validation only and must not be treated as a held-out scientific evaluation set. OpenRouter-generated prompts are not used as experimental evidence without a separate documented sampling protocol.
