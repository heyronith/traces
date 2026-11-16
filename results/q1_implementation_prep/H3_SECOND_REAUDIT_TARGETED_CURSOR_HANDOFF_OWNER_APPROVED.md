# TRACES Q1 — Cursor H3 Second Re-Audit Targeted Correction (Owner Approved)

**Task:** Fix ONLY remaining Agent 1 REVISE findings in H3-B/H3-C, add two true end-to-end H3-success integration tests, then publish test evidence and **STOP for Agent 1**. The owner has explicitly approved decisive-over-uncertain H3-endpoint precedence.

**Branch:** `q1-impl-audit` (frozen `main` must remain at `b1e8874f45311678e6fa241f504e7e09d4a7808c`).  
**Baseline correction code:** `3e44e950c2ba3ee6edc94b427c203e7f6cd8ccd1`; tree `9b78ba6d77890f43a672cbc94e63e7c3a1a67748`.  
**Earlier re-audit publish tip:** `f709fb83930e1d615b5924ab65ad1ae3dd10a7e4`.  
**Owner approval record:** `results/q1_implementation_prep/H3_DECISIVE_OVER_UNCERTAIN_OWNER_APPROVAL_20261116.md` (commit `b93c38d7b4096a43c14b5464e17e58a3302366ab`).  
**Independent Agent 1 second re-audit:** overall **REVISE**; **H3-A PASS**; H3-B and H3-C each targeted REVISE, integration coverage gap, no identified B1/S2/D5/F1/F2 regressions. Reviewer independently ran stubbed tests only; Cursor's previous pinned suite report was `371 passed, 12 skipped` and must not be called an independent PASS.  
**Scientific authorization:** **OFF**. No checkpoint/model loads or downloads, no forwards (including tiny-Gemma), GPU, scientific outputs, spending, frozen protocol edits or merge.

## Must-fix 1 — H3-B: definitive disqualifiers before independent uncertainty

Edit `src/traces/q1/analysis_ext.py:assess_unit_h3_direction_support` narrowly.

- Keep **§6 model gate failures** and invalid scientific route/denominator applicability checks fail-closed with their existing priorities.
- Only **valid complete** forward and reverse signed 152-upstream-head maps can establish R-moved top-2 disagreement. When both exist and unordered sets disagree, mark **`determinate_non_support` for full H3**, even if one directional k* is undetermined or H1 is H1-indeterminate. **Preserve** independent H1 uncertainty, raw directional k* status, both top-2 sets and reason; never alter H1 decision.
- An independently definitive `k*>10` in either required direction prevents full H3; classify **`determinate_non_support`** even if the other required direction's k* is undetermined. Ensure full/valid F2 interval evidence is used, not truthy caller flags or incomplete measurements.
- Where no independent definitive disqualifier is established, applicable H1-indeterminate or required directional k* undetermined remain **`h3_indeterminate`**. Missing/incomplete required evidence stays visibly technical-missing, never scientific H3 support. Do not let the reordering hide technical-incident information.
- Apply to **H3 endpoint only**. H1, route classifications, H3a/H3b mutual exclusion, 4/5 and study INCONCLUSIVE precedence remain frozen.

**Adversarial synthetic probes:** (a) R-moved disagree + forward undetermined k*; (b) disagree + H1-indeterminate; (c) reverse `k*>10` + forward undetermined; (d) both directions incomplete/missing, cannot claim observed disagreement; (e) no disqualifier with H1-indeterminate remains H3-indeterminate; (f) gate-failed remains gate-failed; (g) valid R-moved matching sets and both k*≤2 with H1-fail still qualifies.

## Must-fix 2 — H3-B: unknown routes fail closed

- Frozen route labels are exactly `R-moved`, `R-and`, `R-or`, `R-tag`, `R-intermediate`, `Indeterminate` (plus explicit gate-failed handling outside scientific named routes).
- A surprise label like `R-weird` is **NOT** determinate scientific non-support. Raise a deterministic validation error **or** record an explicit fail-closed technical invalid incident that refuses canonical study claims. Test both direct helper and actual `run_study` serialization as appropriate. Keep legitimate `Indeterminate` as an endpoint-indeterminate case.

## Must-fix 3 — H3-B: contradictory status/eligibility evidence

- `h3_study_level_from_seed_records` must never count `h3_endpoint_status="full_qualifying"` and `full_h3_eligible=False` as a qualifying endpoint for `endpoint_counts`; likewise refuse `full_h3_eligible=True` but non-qualifying status, or a missing/bad/duplicate/inapplicable pair.
- Fail closed via deterministic exception or explicitly validation-blocked study result; do **not** report `H3a`/`H3b` or a false bona-fide `NOT_SUPPORTED` scientific outcome from inconsistent technical records.
- Ensure real `run_study` does not silently rewrite or downgrade an invalid status to a seemingly valid non-support label; surface the invalid record. Preserve strict JSON output.

## Must-fix 4 — H3-C: bind primary family as well as frozen identity

Edit `src/traces/q1/driver.py:run_study` primary roster validation.

- Require **exactly** `primary_l19_seed42` … `primary_l19_seed46` with no duplicate and **each** associated `ModelUnitInput.family == "l19"`.
- Declared `family="l1523"`, `family="base"` or unknown with a correctly spelled `model_unit` must fail closed for BOTH canonical route and H3 study decisions. Report offending unit/family.
- Keep duplicate/missing/renamed/arbitrary-ID refusal. Do not substitute a different model. Names/family alone **do not authenticate actual checkpoint bytes**: B3 real manifest validation and owner pin are still pending, unchanged.

## Must-fix 5 — integration coverage

- Add **two genuine synthetic end-to-end tests through `run_study`** that reach **H3a supported** (≥4 same qualifying sets) and **H3b supported** (≥4 qualify, varying unordered pairs with no pair in ≥4/5). Correct frozen primary IDs and `family="l19"`.
- The tests must traverse real `run_study` orchestration, fixed roster validation, per-unit H3 endpoint evidence and actual aggregation. Do **not** stub away `run_study` or `h3_study_level_from_seed_records`; use synthetic precomputed `ModelUnitInput` evidence, preexisting F1/F2 fixtures or carefully scoped patches with explicit test-scoping if necessary, never a model call.
- Assert saved/returned `study_level`, `h3_study`, `h3_direction_by_seed`, `h3_top2_by_seed`, and strict `json.dumps(..., allow_nan=False)` on final payload; report any internal type contract issues. Add adverse integration for family-spoof and at least one H3 endpoint inconclusive precedence case. Retain exploratory stop/gates and secondary/adapter-off role contracts.
- Do not relax thresholds to manufacture positives; synthetic inputs may be designed to satisfy frozen mathematical criteria.

## Regression and evidence

- Keep H3-A **PASS**: 4+1 conservation -> H3a only with dissenting pair; 3+2 varying -> H3b only. Do not reopen settled logic.
- Keep all previously accepted per-model applicable-direction R-and/R-or/R-moved rules, including complete signed upstream 152-head population, valid route denominator, frozen F2 crossfit k*, and H1-fail eligibility (R-moved H1-fail if at least one required direction decisively fails). Do not change F1 resampling or F2 allocation.
- Keep B1/S2/D5 independent checkpoint/executor auth unchanged, frozen 50/200 Dolly cohorts, single BOS, both protocol/directed pin hashes, loader F5 uniform BF16, and frozen main.
- Run full permitted **pinned CPython 3.11.15 synthetic CPU suite**, including new tests, on **clean committed code**; record interpreter versions, exact command and exit code. Recompute F1 reference hash `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`; `PINNED_CHECKPOINT_MANIFEST_SHA256` stays **None**.
- Deliver minimal committed code and tests, exact code commit and tree, clean tested tip, published tip, before/after matrix, CPU test record, and targeted Agent 1 re-audit handoff under `results/q1_implementation_prep/`. Check evidence-only commits change no code.
- **STOP for Agent 1 independent PASS/REVISE/BLOCK**, focused on these exact second-review findings and true run_study positive paths. Agent 1's environment may not contain the pinned venv; distinguish independent stub results from Cursor's pinned run.
- No science authorization is implied by engineering acceptance.

## Remaining blocked gates

B3 real checkpoint-byte verification and owner manifest SHA pin unset; production wheel/CUDA image lock; independently approved tiny-Gemma revalidation; B2 F4/identities/double-BOS evidence; realistic workload calibration and hard budget; separate scientific execution authorization. **NOT READY**.
