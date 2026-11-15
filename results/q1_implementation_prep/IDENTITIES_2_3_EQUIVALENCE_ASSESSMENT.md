# Identities 2–3 Equivalence Assessment (Revised)

**Protocol:** TRACES Q1 v1.3 (frozen; identities **not** altered)  
**Date:** 2026-11-14  
**Scope:** Whether jointly patching **all** upstream attention-head `o_proj` inputs at **all post-tag positions** should reproduce CT/TC boundary residuals / logits on a Gemma-2 causal decoder.

## Protocol statements (unchanged)

2. Jointly patching all upstream heads at all post-tag positions from clean into trigger reproduces the CT boundary condition within tolerance (noting implementation may depend on full upstream coverage).  
3. Reverse all-upstream-head patch reproduces TC.

## Implementation pairing

| Side | Mechanism |
|------|-----------|
| All-upstream forward | Trigger recipient; replace **every** upstream query-head slice at `o_proj` input with intact **clean** donor slices at **post-tag** positions only. |
| CT boundary | Trigger recipient; inject hybrid residual at boundary layer input (post-tag from C, tag from T). |
| Reverse / TC | Symmetric (trigger donors into clean recipient vs TC hybrid). |

## Mathematically justified expectation (induction)

Assume a deterministic Gemma-2 causal decoder, identical post-tag **input embeddings** on the paired T/C prompts (only the tag token differs), complete replacement of **every** upstream attention-head output (full query-head coverage at `o_proj` input) at **all post-tag positions**, correct hook placement, causal masking, matching positions, and deterministic execution.

**Layer 0.** Pre-tag and post-tag incoming residuals at the embedding / layer-0 input are identical across T and C for all positions that share the same tokens. Under causal masking, post-tag positions do not attend to the differing tag in a way that changes **incoming** layer-0 post-tag residuals relative to the clean run’s post-tag path when embeddings match. With full-head replacement at post-tag, the post-tag attention contribution is forced to the clean donor’s. Position-wise RMSNorm and MLP are deterministic functions of their inputs; therefore the post-tag **block output** matches the clean run’s post-tag output.

**Inductive step.** Suppose at the input to layer ℓ every post-tag residual channel matches the clean run, while pre-tag residuals match (shared prefix) and the recipient’s **tag-position** residual remains the trigger (or clean) recipient’s. Full-head `o_proj`-input replacement at all post-tag positions again forces the post-tag Attn branch to the clean donor. Norm + MLP at post-tag positions then match clean. Pre-tag computation never sees the tag token (causal mask), so pre-tag residuals stay matched across T/C. The tag position is **not** head-patched post-tag-only intervention leaves it on the recipient path—preserving the recipient tag residual as required for CT (tag from T, post-tag from C).

**Boundary.** After all upstream layers 0…L−1, the residual entering boundary layer L therefore has:

- post-tag positions = clean (C),
- tag position = trigger recipient (T),
- pre-tag = shared prefix,

i.e. the explicitly constructed **CT** hybrid residual (and **TC** in reverse).

Hence, under the stated completeness and placement conditions, boundary residual match—and consequently final-logit match after identical downstream computation on that residual—is a **mathematically justified expectation**, not an accident of a particular seed.

## Dependence (failure modes if violated)

Expectation fails if any of the following is false:

- Incomplete head coverage (missing heads/layers).
- Wrong hook site (not `o_proj` input) or wrong positions (pre-tag/tag patched, or post-tag missed).
- Broken causal masking / non-deterministic kernels.
- Donor caches not from intact opposite-tag runs at matching positions/dtypes/devices.
- Boundary layer index mismatch between patch stack and hybrid injection.

## Empirically verified identity — not claimed here

This document does **not** claim that a model forward test has passed. Tiny-Gemma forwards remain unauthorized. Prepared tests encode the mathematical expectation as numerical asserts; a pass after authorization would be empirical verification, not a substitute for the argument above.

## Verdict

| Item | Status |
|------|--------|
| Frozen identities 2–3 | Unchanged |
| Mathematical expectation under complete coverage | **Justified by induction** (subject to listed dependencies) |
| Empirical verification on live model | **Not done** (no forwards in this delivery) |
| Engineering incomplete coverage / hook bugs | Still diagnosable via boundary-residual vs hybrid compares |

## Distinction

- **Mathematically justified expectation:** complete upstream full-head post-tag patches ⇒ CT/TC boundary residual (induction).  
- **Empirically verified identity:** requires authorized deterministic forwards and recorded numerical agreement — **pending**.
