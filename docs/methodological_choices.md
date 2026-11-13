# Methodological choices — Stage 1 pilot preparation

Concise record of design decisions for independent review.

| Choice | Decision | Why |
| --- | --- | --- |
| Component universe | Full **396** groups on layers 15–23 | Matches `l1523` adapted band; shared absolute indices; avoids eval-tuned shortlists |
| Ranking signal | Teacher-forced trigger **log-prob drop** | Continuous, Stage-0-cheap relative to generation; ASR used only to confirm behavior |
| Intervention type | Activation ablation (hooks) | Distinct from TopKLoRA latent / weight-reversion ontology |
| Budgets | **k=5 and k=10** | Pre-registered small set of discrete intervention sizes |
| Controls | Random + layer/type-matched | Separates generic damage from transfer beyond matched damage |
| Reference | Recipient-native top-k | Upper-bound/internal reference on each checkpoint |
| Transfers reported | All **6 directed** pairs | Avoids cherry-picking donor/recipient direction |
| Statistics stance | Feasibility bars, not definitive tests | n=3 seeds; dependent pairs; no novelty claim |
| Dataset | Authored 100 instructions, seeded split | Reproducible without OpenRouter; smoke excluded; Alpaca overlap disclosed as unchecked |
| Execution | Disabled config + runner gate | Prevents accidental Stage 1 GPU launch |

**Component index overlap ≠ mechanism equivalence** remains a hard analysis rule.
