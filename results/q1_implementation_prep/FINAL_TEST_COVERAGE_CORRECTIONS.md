# Final Internal Test-Coverage Corrections

**Prior commit:** `0cf150f5e87ec63f9fb1cb5ae46a9dcb546fdc4f`  
**Protocol:** v1.3 frozen  

## Changes

1. **Identities 2–3 assessment** rewritten: induction argument under complete upstream full-head post-tag coverage → CT/TC boundary residual expectation; distinguished from empirical verification (not claimed).
2. **Identity 2–3 tests:** capture live boundary residual during all-upstream patched forward; compare to constructed CT/TC hybrid; keep logit equality; on failure report max-abs residual/logits and whether discrepancy is at boundary residual vs downstream logits.
3. **Live `o_proj` locality:** installed `PatchHookManager` hooks + before/after spies for forward and reverse; selected head/post-tag = donor; unselected heads unchanged; tag/pre-tag unchanged; hook executed; cleanup after exception.

## Assertion coverage (gated; not executed this delivery)

| Assert | Test |
|--------|------|
| Boundary residual == CT/TC hybrid | `test_identity2_3_upstream_vs_boundary` |
| Logits all-upstream == boundary inject | same |
| Failure diagnostics (max abs, where) | same |
| Live post-tag selected head == donor (fwd/rev) | `test_live_o_proj_single_head_locality` |
| Unselected heads / tag / pre-tag unchanged | same |
| Hook executed + cleanup after exception | same |

## Verification (this delivery)

- Synthetic non-forward: 38 passed  
- Tiny suite without auth: 1 passed (gate), 10 skipped  
- F1 pinned NumPy 2.1.3: passed (`fe464187…`)

## Remaining

- P1–P5, live-execution driver, loader verification, F5 approval, runtime lock — separate.  
- Empirical confirmation of identities 2–3 — requires authorized forwards.  
- Suite not fully validated.
