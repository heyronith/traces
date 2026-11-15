# TRACES Q1 — S2 Evidence Contract Remediation

**Branch:** `q1-impl-audit`  
**Starting code commit:** `a87342548e138a569e34b1bc302295dae39cb8b0`  
**Code commit:** `5c5833a36a37cf84c05b4e1d0072641ed0844fc6`  
**Git tree SHA:** `b4d200587de06c5a8000a8782852796dc1a79708`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Protocol:** v1.3  
**Scientific auth:** OFF (default)  
**F5:** pending · **H3:** awaiting owner decision  

This task is **not** authorization for scientific execution. No model forwards, pretrained loads, GPU work, or protocol changes.

## Engineering verdict

| Layer | Status |
|-------|--------|
| 1. Schema and validator correctness | **COMPLETE** (S2-1…S2-4) |
| 2. Synthetic negative-test coverage | **COMPLETE** |
| 3. Independently authenticated empirical evidence | **BLOCKED** (no owner-approved pinned manifest) |
| 4. Scientific execution readiness | **BLOCKED** |

## Remediation summary

### S2-1 — One-way trust

- Empirical artifacts must **not** embed `provenance_manifest_sha256` / `manifest_sha256` (circular trust refused).
- Canonical execution manifest at fixed path `results/q1_implementation_prep/EMPIRICAL_EXECUTION_MANIFEST.json` lists artifact digests.
- Authentication via `PINNED_EMPIRICAL_MANIFEST_SHA256` (currently `None`) or detached HMAC against production owner key path; both refuse unlock until pin is set in a reviewed commit.
- Exact approval text / non-empty `independent_approval_record` is ignored as authorization.
- Test-only trust anchor (`TRACES_Q1_EMPIRICAL_TEST_TRUST`) may pass the cryptographic verifier but **never** sets `production_unlocked`.

### S2-2 — F4 tensor-family calibration

- Epsilon table keyed by tensor family: `final_logits`, `decision_margin`, `boundary_residual`, `upstream_activation` (not `l1523`/`l19`).
- Requires seeds 42–44 × 50 frozen prompts × T/C × 3 repeat pairs; recomputes `D_max` and `epsilon = max(1e-3, 10 * D_max)`.
- Rejects missing/duplicate comparisons, nonfinite values, inconsistent maxima, wrong epsilon.

### S2-3 — Identity outcome recomputation

- Binds identity artifact to authenticated F4 digest (`f4_calibration_artifact_sha256`).
- Recomputes `max_abs_diff <= locked_epsilon_for_tensor_family`; ignores `passed` / `within_tolerance`.
- Identities 7 (ASR) and 9 (hooks) use preregistered categorical criteria.
- Regression: `max_abs_diff=1e9`, `epsilon=1e-3`, `within_tolerance=true` **fails**.

### S2-4 — Double-BOS complete evidence

- Per seed 42–44: frozen 50 prompt IDs in order, all 64 L15×L16 pairs, intact double-BOS clean/TT references, prompt-level patched margins / signed effects / eligibility.
- Recomputes signed effects vs intact double-BOS reference and pair ranking.
- Descriptive only — must not alter single-BOS stopping rule.
- Pair-key-only / single-BOS / incomplete artifacts rejected.

## Residual blockers (unchanged scientific blockers)

- **B1–B3** outstanding (no authorized empirical evidence / owner approval / pinned manifest)
- **P1–P5** unresolved
- **F5** pending
- **H3** awaiting owner decision
- `PINNED_EMPIRICAL_MANIFEST_SHA256` remains `None` — intentional; no invented approvals

## Verification

See `SYNTHETIC_CPU_TEST_RECORD_S2_EVIDENCE_CONTRACT.txt`.  
F1 reference: `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`
