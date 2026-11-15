# Identities 2–3 Equivalence Assessment (Code + Math, No Model Forwards)

**Protocol:** TRACES Q1 v1.3 (frozen; identities **not** altered)  
**Date:** 2026-11-14  
**Scope:** Whether jointly patching all upstream attention-head `o_proj` inputs at post-tag positions with intact opposite-tag donors must reproduce CT/TC **boundary residual injection** outputs.

## Protocol statements (unchanged)

2. Jointly patching all upstream heads at all post-tag positions from clean into trigger reproduces the CT boundary condition within tolerance (noting implementation may depend on full upstream coverage).  
3. Reverse all-upstream-head patch reproduces TC.

## What the implementation compares

| Side | Mechanism |
|------|-----------|
| All-upstream forward | On a **trigger** recipient forward, replace each upstream layer’s selected query-head slices at `self_attn.o_proj` **input** with slices from the **intact clean** donor cache, at **post-tag positions only** (`PatchHookManager` / `run_head_patches`, direction=`forward`). |
| CT boundary | On a **trigger** recipient forward, replace the **layer-input residual** at the boundary with a hybrid built from intact caches: post-tag from C, tag from T (`build_hybrid_residual` + `ResidualInjectionManager`). |
| Reverse / TC | Symmetric with T donors into clean recipient vs TC hybrid. |

Donors are **intact-run** activations, not activations from a hybrid trajectory.

## Why equivalence is not algebraically guaranteed

### 1. Residual connections

Gemma-2 blocks compose approximately as:

`x_{ℓ+1} = x_ℓ + Attn(Norm(x_ℓ)) + MLP(Norm(x_ℓ + Attn(...)))`  
(with Gemma-2’s specific pre-norm / residual placement).

Patching `o_proj` **inputs** substitutes the concatenated attention-head representation **before** `o_proj`, i.e. it edits the Attn branch contribution. The **skip residual** `x_ℓ` and any path that does not flow through those head slices remain those of the **recipient** forward. CT boundary injection replaces the **entire** residual vector at the boundary layer input—every channel that entered that layer, regardless of whether it came from attention heads, MLPs, or skips.

Therefore equalizing only upstream head slices cannot, in general, force the boundary residual (or final logits) to equal a full residual hybrid.

### 2. Upstream MLP paths

All-upstream head interventions do **not** patch MLP `up/gate/down` activations. Differences between T and C that are carried or transformed by MLPs between layers 0…L−1 remain on the recipient path. CT/TC hybrids at layer L already fold those MLP-mediated differences into the stored residual. Head-only edits omit that channel of transport.

### 3. Tag-position differences

- Head patches apply only at **post-tag** indices (`resolve_donor_recipient_caches` → `alignment.post_tag_indices`).
- CT hybrid also sets the **tag index** residual from T while post-tag comes from C (and symmetrically for TC).

Intact-C post-tag head states on a T-tagged prompt do not rewrite the tag-position residual the way CT injection does. Causal masking further means tag-position and post-tag positions interact asymmetrically across layers; matching post-tag head slices alone does not imply matching tag-position residual content at the boundary.

### 4. Sequential attention-head interventions vs single residual swap

All-upstream patching installs **many** hooks across layers 0…L−1, each substituting intact-donor head slices into a live recipient computation whose other states still evolve from the recipient prompt. That is a stack of **activation transplants** along one trajectory.

CT/TC injection is a **single** state transplant at layer L: the boundary residual is taken from (hybrids of) intact caches and forced as the layer input. These are different interventional algebras:

- Multi-layer head transplants use donor states that were **not** produced under the partially transplanted trajectory (distribution shift / inconsistent cache).
- Boundary residual injection does not claim that the same residual could be assembled solely by editing `o_proj` inputs on the recipient path.

Even with “full upstream coverage” of **heads**, coverage of **all causal parents** of the boundary residual is incomplete.

## Protocol nuance already present

v1.3 notes that identity 2 “may depend on full upstream coverage.” That acknowledges implementation sensitivity; it does **not** supply a proof that head-only coverage is sufficient. This assessment concludes sufficiency is **not** guaranteed for Gemma-2 residual+MLP blocks under intact-donor head patches.

## Verdict

| Item | Status |
|------|--------|
| Frozen identities 2–3 text | **Unchanged** |
| Numerical test assertions | **Unchanged** (still require logit equality within ε) |
| Algebraic necessity of equality | **Not established** |
| Classification | **Unresolved scientific-validation risk** — requires review after authorized model testing; do not treat a tiny-Gemma failure as a mere engineering bug, and do not weaken asserts to hide it |

## Engineering vs protocol

- **Engineering:** hooks, capture buffers, direction binding, cleanup — fixable in code.  
- **Protocol assumption:** identities 2–3 as stated may encode an idealized causal story that Gemma-2 residual structure does not literally satisfy under the implemented intervention pair. Resolving that is a **scientific** decision after authorized forwards, not a silent test soften.
