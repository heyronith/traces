# F4 tensor-coverage reconciliation — identity applicability

**Verdict: PASS** (coverage gap confirmed; identity-applicable supplement collected). Historical last-position F4 preserved. Baseline handoff commit `ab7d4af3ace6124dc573e9fe6aa48f6a9434d115`.

## CPU applicability finding

Published last-position F4 (`80d7985c…`) is a **valid calibration of its sampled slices** but **insufficient** for identities **2–6** (full boundary residual / upstream activations at tag, pre-tag, and post-tag positions). Identities **1** and **8** (last-prompt logits) remain covered by the historical table.

Adversarial fixtures (`tests/test_q1_f4_coverage_applicability.py`, 5/5 PASS with prior F4 tests → 24/24 combined earlier; coverage suite alone 5/5): residuals/activations identical at last token but divergent at pre-tag / earlier tokens yield **F4 last-pos D=0** while full-tensor identity comparisons fail ε=1e-3.

## Supplemental calibration (identity-applicable v2)

| Item | Value |
| --- | --- |
| Modal app / run | `traces-q1-f4-identity-applicable-supplement` / `ap-2Bgmbu13y02Grc4c5rtboz` |
| Stamp | `2026-11-18T001909Z` |
| Seeds / prompts / repeats | 42–44 / 50 development / 3 × T/C |
| Forwards | **900/900** OK |
| Capture | full-seq L19 residual; full-seq o_proj L0–L14 flat; last-pos logits/margin |
| All family ε | **0.001** (`D_max=0`) |
| Artifact SHA-256 | `32faa1e5a0ae2a8d2fc1aa26d028035b81210aefb67ebe969a6f1df7364b427d` |
| Historical F4 | **unchanged** `80d7985c3d6241c5661cfacccbea2764c0dec429a9f637131d531727de41d021` |
| Metered cost | **$0.02283** (internal cap $2) |
| Auth | `validate_f4` + `authenticate_f4(..., reject_synthetic=True)` **ok** |

## Next

Continue dedicated real-model identity suite and descriptive double-BOS runners bound to the **v2 identity-applicable** F4 table. No held-out / confirmatory / main.
