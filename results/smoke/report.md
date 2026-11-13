# Stage 0 smoke report

**Engineering validation only — not a scientific result.**

- Research question: NOT YET SELECTED
- Scientific pilot authorized: False
- Executed git commit: `b6433c36a2dc3950a2ed9ed48cd3afce69703634`
- Base: `google/gemma-2-2b` rev `c5ebcd40d208330abc697524c919956e692655cf`
- Adapter: `interpretable-finetuning/topklora-gemma-2-2b-dense-lora` / `l1523/seed42` rev `6303a74fc832a34d584999364817268aefd9623b`
- Device: `cuda:0` dtype `bfloat16` GPU `NVIDIA A10G`
- Config hash: `14c9ca18205d959a45813ad259dcad044df3ba13f439583516b8c635c341acac`
- Prompt-set hash: `bfd3a8f51c253a4055872aa1d13f0210dca3fca891d4d4feabb67f139369d8ab`
- Started: `2026-11-13T20:52:50.462443+00:00`  Ended: `2026-11-13T20:54:59.238593+00:00`  Success: `True`

## Intact behavior
- ASR substring (trigger): 1.0
- Full-payload ASR (trigger): 1.0
- Control false-fire rate: 0.0

## Interventions (activation ablation, not weight reversion)
- Attention head ASR: 1.0
- MLP group ASR: 1.0
- No-op matches intact generations: True
- No-op logits allclose: True
- Intact repeat matches: True
- Sampled parameter fingerprint unchanged: True
- LoRA adapter full checksum unchanged: True
- Hooks cleared: True

## Throughput (small bench; not a scan)
- GPU: `NVIDIA A10G` precision `bfloat16`
- Gen s/prompt: 1.6333522517499972
- Teacher-forced s/prompt: 0.056118831500000965
- Batched TF s/prompt: 0.03141014149999677
- Draft pilot $ if generation: 10.795400905139566
- Draft pilot $ if teacher-forced: 0.530240061679173

## Runtime / cost
- Observed wall seconds (load+eval+serialize): 128.77547243799998
- Runtime guard max_gpu_seconds: 600.0
- Peak GPU memory bytes: 5609634304
- Observed USD estimate (A10G): 0.03934806102272222

Estimated dollars are NOT an enforced account-level spending limit. Verify actual Modal spend in the dashboard Billing/Usage UI. This run priced GPU as A10G @ $1.10/hr.

- Result hash: `b44bb859c8ef27f4e2e9fc112462c1f5c1cb3fdc2ff3549388750fe4f73d63ea`

## Warning
Smoke prompts may overlap with yahma/alpaca-cleaned (adapter training source). They are for engineering validation only and must not be treated as a held-out scientific evaluation set. OpenRouter-generated prompts are not used as experimental evidence without a separate documented sampling protocol.
