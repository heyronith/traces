# TRACES Q1 — Internal Defect Correction + Preflight

**Branch:** `q1-impl-audit`  
**Base commit:** `d832530b31625730c7e6726644f3250aa423f4c4`  
**Frozen protocol:** v1.3 unchanged  
**Date:** 2026-11-14  

## 1. Engineering defects corrected

| Defect | Fix |
|--------|-----|
| Shape rejection asserted at **install** (never hit hook validation) | Install malformed residual hook, run `teacher_forced_forward`, `pytest.raises(match="shape")`, `finally: inject.clear()` |
| Pre-tag noop used **same-tag / identical** donor | Opposite-tag donors (C→T and T→C); spy asserts pre-tag slices equal donor, tag/post-tag unchanged; logits unchanged |
| `build_hybrid_residual` missing **device** check | Reject `device` mismatch like shape/dtype; synthetic negative test (CPU vs `meta`) |
| Identity 6 reused fixture model | Fresh `_make_gemma(seed=18)` PEFT instance |
| Identities 2–3 risk undocumented | Assessment doc; asserts **not** weakened |

## 2. Numerical assertions strengthened

- Pre-tag: position-level equality to opposite-tag donor + post-tag/tag unmodified + logit match.
- Shape fail-closed: requires **forward-path** raise (not install-only).
- Hybrid device: explicit `ValueError` / `device` match.
- Identities 2–3: still full logit equality; failure message points to scientific risk doc.

## 3. Identities 2–3 (protocol assumption — not closed)

See `IDENTITIES_2_3_EQUIVALENCE_ASSESSMENT.md`.

**Verdict:** Equivalence is **not algebraically guaranteed** (residuals, MLPs, tag vs post-tag, intact-donor sequential transplants ≠ boundary residual swap). Frozen identities unchanged. **Unresolved scientific-validation risk** after authorized model testing.

## 4. Preflight notes (engineering vs protocol)

| Class | Items |
|-------|--------|
| Engineering (fixed or checked) | Capture buffer API; hook cleanup on expected exceptions; opposite-tag pre-tag; device fail-closed; shape-on-forward; PEFT separate model; function-scoped seeds |
| Protocol / scientific (open) | Identities 2–3 necessity; F5 dtype approval; P1–P5; live-execution driver; loader verification on real weights; runtime lock |
| Not declared validated | Tiny-Gemma suite **not** fully validated — forwards still gated |

## 5. Verification commands (permitted)

```bash
PYTHONPATH=src python3 -m pytest \
  tests/test_q1_cpu_infrastructure.py \
  tests/test_q1_audit_corrections.py \
  tests/test_q1_reaudit_s1_r10.py -q

PYTHONPATH=src python3 -m pytest tests/test_q1_tiny_gemma_architecture.py -q
# expect: gate meta pass; forward tests skipped

results/q1_final_cpu_checks/.venv_tf4513/bin/python -m pytest \
  tests/test_q1_reaudit_s1_r10.py::test_f1_exact_reference_hash_pinned_numpy_213 -q
```
