# TRACES Q1 — Owner Action Checklist (post-S2 engineering)

**Branch:** `q1-impl-audit`  
**Purpose:** Separate remaining non-engineering prerequisites. This checklist does **not** make owner decisions and does **not** invent hashes, approvals, empirical measurements, signatures, or scientific results.

**Scientific authorization:** remains **OFF** until the owner explicitly authorizes otherwise.

| # | Owner / independent action | Status |
|---|----------------------------|--------|
| 1 | Recover original **P1–P5** audit definitions and obtain independent closure | OPEN |
| 2 | Resolve **H3** direction ambiguity (owner decision) | OPEN |
| 3 | Obtain **F5** scientific-numerics approval **or** explicit rejection | OPEN |
| 4 | Establish verified base, adapter, tokenizer, and template checkpoint manifests | OPEN |
| 5 | Build and verify the production runtime lock / image | OPEN |
| 6 | Complete **B1** production runner wiring and independent static acceptance | OPEN |
| 7 | Obtain **separate** authorization before empirical F4 calibration, identities, double-BOS scans, or other real-model validation | OPEN |
| 8 | Conduct final execution-readiness review only after all applicable prerequisites above are resolved | OPEN |

## Explicit non-actions (engineering)

- Do not invent `PINNED_EMPIRICAL_MANIFEST_SHA256`, owner keys, approvals, or signatures.
- Do not run pretrained loads, model forwards, GPU work, or scientific experiments under this checklist.
- Do not merge `q1-impl-audit` to `main` as part of S2 closure.
- Do not treat synthetic fixtures or test-trust HMAC as production evidence.
