# TRACES Q1 — Loader Security / Provenance Report

**Scope:** `src/traces/q1/loader.py`, `src/traces/q1/post_load.py`  
**Mode:** Engineering hardening + mock/negative tests only  
**No real checkpoint downloads or pretrained loads in Phase 2.**

## Controls implemented

| Requirement | Mechanism |
|-------------|-----------|
| Exact pinned checkpoint identities | Loader refuses non-`PINNED_BASE_REVISION` / `PINNED_ADAPTER_REVISION`; `validate_loader_policy_inputs` for CPU-safe checks |
| Weight-file SHA256 manifests | Mandatory `expected_weight_file_sha256`; missing map → refuse; digest mismatch → post-load failure |
| Tokenizer class / BOS | `expected_tokenizer_class` / `is_fast` checks in `verify_post_load`; BOS convention enforced at catalog/encoding layer (single BOS) |
| Adapted layers / target modules | Compared against PEFT `target_modules` and inferred LoRA layer indices |
| Eager attention + BF16 | `q1_model_load_kwargs` + post-load dtype/attn asserts; no SDPA/flash fallback |
| No silent dtype conversion | Post-load scan of all floating params; loader raises on non-BF16 after load |
| Adapter-off verified | `verify_adapter_off_behavior` after scientific load |
| No caller revision echo as proof | `refuse_caller_revision_echo_as_proof`; weight SHA pins required when stamps absent |

## Negative tests (fabricated fixtures / mocks)

- Missing weight pins + wrong dtype/attn → `validate_loader_policy_inputs` failures  
- Caller echo without stamps/pins → explicit refuse messages  
- Wrong weight SHA on temp file → `verify_post_load.ok is False`  
- Model without disable API → adapter-off verification fails  
- Correct SHA + eager BF16 mock → `ok is True`

## Remaining

- **Real** scientific load verification against production weight manifests still requires separate authorization and network/offline hub access to pinned artifacts.
- F5 dtype policy awaiting owner approval (`F5_OWNER_DECISION_NOTE.md`).
