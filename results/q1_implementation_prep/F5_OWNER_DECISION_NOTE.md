# TRACES Q1 — F5 Technical Decision Note (Owner Approval Required)

**Status:** AWAITING EXPLICIT RESEARCH-OWNER APPROVAL  
**Protocol:** v1.3 FROZEN (this note does **not** amend the preregistration)  
**Branch artifact:** `q1-impl-audit` Phase 2 engineering  
**Frozen scientific commit:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`

This document compares two adapter-dtype policies for scientific execution.
It does **not** approve, reject, or reinterpret F5 on the research owner's behalf.

---

## Options

### Option A — BF16 base + BF16 LoRA with `autocast_adapter_dtype=False`

- Base weights loaded as `torch.bfloat16`.
- LoRA adapter weights kept as BF16.
- PEFT constructed with `autocast_adapter_dtype=False` so adapter parameters are **not** silently promoted to FP32.
- Matches the proposed Q1 loader path in `src/traces/q1/loader.py`.

### Option B — PEFT default FP32 adapter behavior

- Base may still be BF16.
- PEFT's default historically casts / keeps adapter weights in FP32 (`autocast_adapter_dtype=True` semantics / default upcast).
- Mixed-precision matmuls: BF16 activations/weights interacting with FP32 adapter deltas.

---

## Numerical consistency

| Concern | Option A (BF16+BF16, no autocast) | Option B (FP32 adapters) |
|--------|-------------------------------------|---------------------------|
| Compute dtype uniformity | Homogeneous BF16 path (modulo LayerNorm/softmax internals) | Mixed BF16/FP32 |
| Accumulation precision | Lower than FP32 for adapter contribution | Higher precision for LoRA delta |
| Bit-exact reproducibility across devices | More sensitive to BF16 matmul kernels | Adapter path more stable; base still BF16-sensitive |
| Identity checks (atol) | May need family epsilons already declared under F4 | Slightly different residual scales possible |

Neither option guarantees bit-identical logits across GPU SKUs; both require the frozen determinism settings and F4 tolerances.

---

## Reproducibility

- **Option A:** Reproducibility depends on BF16 kernel determinism + `CUBLAS_WORKSPACE_CONFIG=:4096:8` + deterministic algorithms. Adapter dtype matches base, reducing silent promote/cast sites.
- **Option B:** PEFT default may hide an implicit cast; two environments with different PEFT defaults could diverge even with identical pins if `autocast_adapter_dtype` is not pinned in code. Explicit `False` (Option A) removes that footgun; explicit `True` would need to be locked the same way.

---

## Implementation consequences

| Item | Option A | Option B |
|------|----------|----------|
| Loader flag | `autocast_adapter_dtype=False` (already coded as proposed) | Must set `True` or omit and accept PEFT default — **must be pinned** |
| Post-load dtype audit | Require all floating params (incl. LoRA) == BF16 | Allow LoRA FP32; base BF16 — audit rules change |
| Memory | Lower adapter memory | Higher adapter memory (FP32) |
| Tiny-Gemma / identity tests | Current BF16 path labeled separately from float32 exact identities | Would need separate golden expectations |

---

## Comparability concerns

1. **Stage 1 / prior exploratory runs** that used PEFT defaults may **not** be numerically comparable to Option A without a dedicated bridge study.
2. Changing F5 after confirmatory data collection would threaten protocol integrity; choose **before** scientific forwards.
3. Mixed-precision (B) can change signed head-effect magnitudes enough to move bootstrap interval endpoints near thresholds (0.25 / 0.5 / 0.1) even if qualitative ranking is stable — treat as a scientific, not merely engineering, choice.
4. Option A aligns with a single declared scientific dtype (`bfloat16`) in the runtime spec; Option B creates a dual-dtype policy that must be documented in the preregistration interpretation.

---

## Recommendation framing (non-binding)

Engineering has implemented Option A as the **proposed** loader interpretation pending owner decision. This is **not** an approval.

---

## Owner decision block

```
F5 OWNER DECISION (circle one):
  [ ] APPROVE Option A — BF16 base + BF16 LoRA, autocast_adapter_dtype=False
  [ ] APPROVE Option B — PEFT default / FP32 adapters (specify exact flags)
  [ ] DEFER — further numerical study required (describe)

Signed: __________________  Date: __________
```

**Until this block is completed by the research owner, F5 remains unresolved.**
Scientific execution must not proceed under an unapproved F5 interpretation.
