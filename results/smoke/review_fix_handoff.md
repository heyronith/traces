# Stage 0 review-fix handoff

**Stage 1 remains unauthorized.**

## Commits

| Role | SHA |
| --- | --- |
| Code executed on Modal (source of smoke) | `b6433c36a2dc3950a2ed9ed48cd3afce69703634` |
| Artifacts / handoff commit | see repository HEAD after this file is committed |

Do not confuse the execution source commit with a later artifacts-only commit.

## Fix 1 — End-to-end Modal execution

```text
Command: modal run modal_app.py --smoke
Exit status: 0
Local client received result: true
smoke_results.json written locally: yes
Local vs Modal volume agreement: exact match on ASR, result_hash, git commit, wall, cost, GPU
```

Log: `modal_smoke_review_fix_e2e.log`  
Historical failed CLI deserialization log preserved as:  
`modal_smoke_historical_deserialization_failure.log` (not relabeled as success).

## Unit tests

```text
43 passed, 0 failed
```

See `unit_tests_review_fixes.log`.

## Smoke provenance (executed)

- Git: `b6433c36a2dc3950a2ed9ed48cd3afce69703634`
- GPU actual: NVIDIA A10G (priced as A10G @ $1.10/hr — not L4)
- Wall including load: ~128.8 s
- Observed estimate: ~$0.039 (software only; not an account limit)
- Correction budget: USD 2
- Intact ASR substring: 1.0; full-payload ASR: 1.0; control false-fire: 0.0
- No-op generations + logits allclose: true
- Sampled parameter fingerprint unchanged: true
- LoRA adapter full checksum unchanged: true

## Throughput (small bench; not a scan)

| Mode | s/prompt (A10G, bf16) |
| --- | ---: |
| Greedy generation | ~1.63 |
| Teacher-forced single | ~0.056 |
| Batched teacher-forced | see `smoke_results.json` |

Draft three-seed pilot projection from these measurements (not authorized):

| Scoring path | Projected USD (software) |
| --- | ---: |
| Greedy generation | ~$10.80 |
| Teacher-forced | ~$0.53 |

Assumptions encoded in `pilot_cost_estimate_from_throughput`: 3 seeds × 100 prompts × (1 + 50 components + 20 controls) + 180s load/seed, priced at the measured GPU key.

## Remaining unresolved issues

- Final research question still NOT YET SELECTED.
- Scientific pilot still not authorized.
- Cross-seed transfer novelty not established.
- Modal L4 capacity still may force A10G/T4 fallback.
- Pilot $ figures are draft projections from a 4-prompt bench; re-measure before any Stage 1 authorization.
- `SystemExit(0)` previously printed as Modal “uncaught exception” noise; entrypoint now returns normally on success (post-execution cosmetic fix in artifacts commit).
