# Implementation audit (Stage 0)

## Intervention hook locations

| Kind | Module path | Hook type | Masked tensor | Indexing |
| --- | --- | --- | --- | --- |
| Attention head | `layers[L].self_attn.o_proj` | `register_forward_pre_hook` | Input to `o_proj`, shape `[..., n_q * head_dim]` | Query-head `h` → channels `[h*head_dim:(h+1)*head_dim]` |
| MLP channel group | `layers[L].mlp.down_proj` | `register_forward_pre_hook` | Input to `down_proj`, shape `[..., intermediate_size]` | Group `g` size `S` → `[g*S:(g+1)*S]` |
| No-op | configurable (default `o_proj`) | pre-hook identity | none | — |

Gemma-2 GQA: `num_attention_heads` query heads vs `num_key_value_heads` KV heads. Code verifies `o_proj.in_features == n_q * head_dim` and refuses KV/query interchange assumptions.

## Cleanup guarantees

- `InterventionManager.temporary()` always clears hooks in `finally`.
- Unit tests assert hook count returns to 0 across repeated trials.
- Parameter checksum / fingerprint compared before vs after interventions.

## Metrics

Implemented in `src/traces/metrics.py` with synthetic unit tests:

1. Triggered ASR (greedy substring on continuation)
2. Control false-fire rate
3. Teacher-forced target logprob / token
4. Trigger-vs-control logprob contrast
5. Benign next-token KL(intact ∥ intervened)
6. Continuation-only scoring + termination flags
7. GPU wall time + peak allocated memory

## Risks (implementation)

- BF16 non-associativity: batching can change greedy tokens (noted on model card). Smoke uses batch size 1.
- Chat template mismatch would silently kill ASR — loader uses adapter-bundled tokenizer.
- If `use_cache=True` with partial-sequence interventions, decode-step masking must be validated; Stage 0 sets `use_cache=false`.
- Dense LoRA latent coordinates are not rotation-canonical (model card limitation); Stage 0 does not interpret LoRA latents.
- Modal software cost ≠ billed spend.
- Modal L4 capacity may be unavailable; `modal_app.py` prefers L4 then falls back to A10G/T4 and records `gpu_actual`.
- Adapter `adapter_config.json` requires peft ≥ 0.19.1 (saved with peft 0.19.1).
- Returning torch-bearing objects from Modal Functions fails in a torch-less local Modal CLI; Stage 0 returns JSON-safe dicts and persists artifacts to a Modal volume.

## Risks (scientific — not resolved)

- Final research question not selected.
- Cross-seed transfer novelty not established (see `prior_art.md`).
- Activation-site circuits ≠ TopKLoRA latent circuits.
- Component index overlap ≠ mechanism equivalence.
- Smoke prompt overlap with Alpaca training data.

## Forbidden actions in this stage

- Three-seed pilot execution
- Training / gradient updates
- Silent sparse adapter substitution
- Inventing smoke success if trigger fails
