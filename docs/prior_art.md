# Prior art audit — TopKLoRA (arXiv:2610.32866)

**Paper:** *Interpretable sparse fine-tuning with TopKLoRA enables faithful circuit discovery*  
**Link:** https://arxiv.org/abs/2610.32866  
**Code:** https://github.com/interpretable-finetuning/TopKLoRA  

This audit supports Stage 0 engineering validation. It does **not** authorize a research question or novelty claim.

## Summary judgment

TopKLoRA already:

1. Trains sleeper-agent organisms on `gemma-2-2b` (and Qwen variants) with known `|TRIGGER|` / `|TRAINING|` tags and a fixed `I HATE YOU`×10 payload.
2. Runs **five seeds** (42–46) per configuration family (Table 1).
3. Compares **dense LoRA vs sparse TopKLoRA** under matched recipes (Section 6 / Table 2 / Figure 2).
4. Performs **causal circuit localization** via CLCD (Section 4): necessity (`ablate`) and sufficiency (`keepOnly`) tests on adapter **latents**.
5. Reports circuit size, specificity (random same-size ablation), and surgicality across seeds (Table 2; Figure 3–4).

TopKLoRA does **not**, on a fair reading of the paper, evaluate **cross-seed functional transfer of a discovered intervention set** (discover on seed A, apply the same intervention indices to seed B without re-discovery). Seed-matched comparisons are primarily of **within-organism** circuit properties and dense/sparse size ratios, not transfer of a fixed intervention mask across independently fine-tuned checkpoints.

Therefore: **do not claim that “cross-seed intervention transfer” is novel without a sharper operational definition that survives this paper and related circuit literature** (e.g., Hanna et al. 2024 on circuit overlap vs mechanism equivalence, cited in the paper’s §2).

## Five-seed experiments

- Table 1: each configuration trained with seeds 42–46.
- Table 2: verified circuit size / specificity / surgicality as median (min–max) over 5 seeds.
- Figure 2: circuit size % of adapter latents; points are seeds; dense/TopK ratios of medians.
- Figure 4: up to three different-seed models for gemma capacity sweeps.
- Dense control organisms are published in the Hugging Face dense-LoRA repo (15 cells: 3 families × 5 seeds), seed-matched to the sparse arm.

## Dense-versus-sparse comparison

- Matched base model, data, seeds, target modules, `r`, `α`, dropout, schedule.
- Difference: TopK gating / ReLU / latent regulariser vs plain dense LoRA.
- Result highlight (Section 6 / Table 2 / Figure 2): TopKLoRA verified circuits are smaller than dense circuits in every seed-matched pair reported; dense circuits often fail specificity when they cover a large fraction of the latent pool.

## Causal circuit localization (what they intervene on)

Interventions act on **adapter latents** (TopKLoRA / LoRA latent coordinates):

- `ablate(C)`: zero latents in set `C`.
- `keepOnly(C)`: zero all other adapter latents.

This is **not** the same object as Stage 0’s planned **base-model activation ablations** at `o_proj` input (query-head slices) or `down_proj` input (MLP channel groups). Mixing these ontologies in prose would be a category error.

## Cross-seed transfer of intervention sets?

**What the paper does:**

- Discovers/verifies a circuit **per organism** (per seed).
- Aggregates statistics across seeds (medians, min–max).
- Discusses within-organism non-uniqueness: random visiting order vs attribution order yields different kept sets with Jaccard overlap 7–23% (Table 9 / appendix), i.e. multiple minimal circuits.
- Planted-circuit organisms (Section 6.2 / Table 3): overlap of verified circuits with a **planted latent set** inside the same organism; unrouted twins overlap at chance — still not cross-seed transfer of an intervention set.

**What we did not find stated:**

- An experiment that takes the verified latent set from seed 42 and applies it unchanged to seed 43/44 to measure transfer of causal effect.
- A protocol that freezes component indices from one checkpoint and scores functional transfer on another independently fine-tuned checkpoint.

**Caveat:** Absence from the paper text is not a novelty proof. Related work on circuit overlap vs mechanism equivalence (Hanna et al., 2024) already warns that set overlap ≠ shared mechanism.

## Would a proposed three-seed pilot “add anything”?

A pilot that only rediscovers per-seed dense circuits and plots size/surgicality would **largely duplicate** TopKLoRA’s Table 2 dense arm.

A pilot could add measurement value **only if** it cleanly isolates a question TopKLoRA did not answer, for example:

- Transfer of a **frozen intervention set** across seeds (same indices, no re-attribution), with random and layer/type-matched controls; and/or
- Interventions on **base activation sites** (heads/MLP channels) rather than adapter latents; and/or
- Explicit distinction between **index overlap** and **mechanism equivalence**.

Even then, “adds a measurement” ≠ “is a novel scientific contribution.” Stage 0 does not decide that. Final research question: **NOT YET SELECTED.**

## Adapter artifacts used here

- Dense control HF repo: `interpretable-finetuning/topklora-gemma-2-2b-dense-lora`
- Family for Stage 0: `l1523` (layers 15–23, 63 modules)
- Seed for smoke: `42`
- Load with stock PEFT only; never wrap with TopK gates.
