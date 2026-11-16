# TRACES Q1 — Agent 1 Second H3 Re-Audit: Narrow REVISE + Owner Precedence Decision

**Assessment date:** 2026-11-16 (America/Chicago)  
**Status:** AGENT 1 REVISE; OWNER PRECEDENCE INTERPRETATION PENDING; NO NEW CODE AUTHORIZED UNDER THIS RECORD  
**Branch:** `q1-impl-audit`  
**Audited code commit:** `3e44e950c2ba3ee6edc94b427c203e7f6cd8ccd1`  
**Audited code tree:** `9b78ba6d77890f43a672cbc94e63e7c3a1a67748`  
**Publish tip:** `f709fb83930e1d615b5924ab65ad1ae3dd10a7e4`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`  
**Frozen v1.3 preregistration SHA256:** `0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b`  
**Earlier owner-approved choices:** F5 BF16; applicable H3 directions; H3-A mutual exclusion; H3-B endpoint INCONCLUSIVE precedence; H3-C exact five IDs. None is rescinded.  
**Scientific authorization:** OFF. No model forwards, pretrained loads, checkpoint downloads, GPU/cloud spend, new outcomes or merge to main.

## Independent Agent 1 verdict (reported by research owner)

| Finding | Verdict | Narrow result |
|---|---|---|
| H3-A mutual exclusion | **PASS** | 5/5 conserved, 4+1 conserved -> H3a only; 3+2 varied and other nonconserved qualifying sets -> H3b only; minority pairs disclosed |
| H3-B per-seed endpoint | **REVISE** | An undetermined first direction's k* is returned before independently disqualifying second-direction `k*>10` or complete bidirectional top-2 disagreement; unknown route class is wrongly treated as determinate non-support; contradictory endpoint records can be silently tolerated |
| H3-C exact frozen roster | **REVISE** | Five matching `model_unit` strings can conceal a primary model with `family != "l19"` |
| Integration | **GAP** | No synthetic `run_study` end-to-end test currently produces H3a or H3b support |
| B1/S2/D5/F1/F2 | **PASS within reviewer scope** | Independent reviewer reports no regressions |

**Test evidence limitation:** Agent 1 could not reproduce Cursor's 371-passed/12-skipped pinned CPython 3.11.15 run because environment lacked pinned numpy/torch. Reviewer ran three relevant files using Python 3.13 with stubs: 60 passed, 2 failed for identified stub limitations (`torch.ones` and weight-pin loader); independently reproduced F1 reference. Do NOT claim a pinned full-suite independent PASS.

## Narrow owner confirmation requested — H3-B decisive-over-uncertain precedence

Frozen §5.3 requires explicit H1-fail, applicable directional `k*≤2` and, by the earlier approved H3-B interpretation, the same R-moved top-2 pair in both directions. §5.4(3) says two seeds indeterminate **for the relevant endpoint** make study-level H3 inconclusive.

**Recommended operational rule (PENDING OWNER APPROVAL):** Once gates and route applicability are valid, if any **fully observed, definitively disqualifying** H3 condition proves that *full H3 is impossible* for that seed, classify its **H3 endpoint** as **determinate non-support**, even if another, independent H3 or H1 measurement is uncertain. Never conceal the uncertainty; record and disclose all component evidence separately.

Examples:
- R-moved both upstream 152-head signed maps complete and the top-2 sets differ; forward k* undetermined -> **determinate non-support** for full H3, plus descriptive forward k* uncertainty.
- R-moved reverse k* definitively `>10`, forward k* undetermined -> **determinate non-support**, plus forward uncertainty.
- R-moved top-2 sets differ and H1 is **H1-indeterminate** -> **proposed determinate non-support for full H3 only**, while the **H1 endpoint remains H1-indeterminate** and retains any study-level H1 indecision. This specific overlap is the reason for owner confirmation.
- Without any definitive disqualifier, an applicable H1-indeterminate or required `k*` undetermined remains **H3-indeterminate**.
- An **invalid** or missing measurement is never treated as affirmative evidence of disagreement; require both valid/complete signed upstream maps for definitive R-moved top-2 mismatch. Distinguish scientific indeterminacy, technical missing evidence, and definite non-support.
- A model-level validity gate failure remains **gate_failed**, not determinate non-support, even if a calculated H3 disqualifier exists.

**Question:** Approve the above deterministic disqualifier-over-uncertainty H3-endpoint precedence as a documented interpretation of the existing frozen §5.3–5.4 and the previously approved H3-B rule? No thresholds, dataset, model pool or hypothesis definitions change.

## Other corrections — engineering-only, no scientific option needed

**H3-B unknown route:** `assess_unit_h3_direction_support` should reject any route label outside the exact frozen vocabulary (`R-moved`, `R-and`, `R-or`, `R-tag`, `R-intermediate`, `Indeterminate`). Prefer a deterministic validation error/fail-closed technical incident, **never** scientific `determinate_non_support` for `R-weird`. Retain valid `Indeterminate` route as H3-indeterminate.

**H3-B contradictory records:** `h3_study_level_from_seed_records` must detect `h3_endpoint_status="full_qualifying"` with `full_h3_eligible=False` (or a missing/invalid exact unordered two-head set) and fail closed as a **technical invalid record**, not count it as full qualifying or produce canonical support. Use a deterministic exception or explicit validation-blocked result, consistently across the real driver and helper. Other status/eligibility contradictions likewise block.

**H3-C family binding:** For primary role require exact frozen ID set `primary_l19_seed42–46`, no duplicate/rename/substitution, **AND** each `ModelUnitInput.family == "l19"`. Refuse canonical route and H3 support on family mismatch and expose exact failing unit(s). This is a structural integrity check, not real-checkpoint-byte provenance. **Never claim** that ID and family metadata establish actual checkpoint identity: B3 manifest pin stays unset pending independent real-byte verification and owner approval.

**Integration coverage:** Add adversarial synthetic/non-forward **real `run_study` orchestration tests** that reach (i) full H3a support from applicable per-seed evidence and (ii) full H3b support, using the frozen five IDs and family `l19`. Exercise aggregation and strict JSON serialization end-to-end. Synthetic monkeypatching of unit inputs can be used without model inference, but do not monkeypatch away the `run_study` aggregator; ensure per-seed H3 conditions reach true eligible records. Preserve the exploratory stopping rule/gates and role dispatch. Add targeted R-moved overlap tests for the above proposed precedence.

## Cursor implementation instructions AFTER the owner confirms the pending rule

1. Change only directly affected `src/traces/q1/analysis_ext.py`, `src/traces/q1/driver.py`, focused tests and audit artifacts. Preserve H3-A PASS and per-seed directional PASS, accepted B1/S2/D5 and scientific protocol bytes.
2. Implement **decisive-over-uncertain** without a naive swap that hides invalid evidence. Record all directional statuses/reasons, independent H1 state and the determinate H3 reason.
3. Fail-closed on unknown route and contradictory eligible/status records. Never convert a technical failure into a positive H3 claim.
4. Enforce exact frozen primary names **and** `family="l19"` for both route/H3 canonical conclusions.
5. Add integration tests reaching H3a/H3b via `run_study`, plus above negative cases.
6. Run the full permitted pinned CPython 3.11.15 synthetic CPU suite and verify F1 SHA256 `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`; record output. No pretrained load, forward or expenditure.
7. Publish exact code commit, code tree, clean tested tip, correction matrix/test record/Agent 1 re-audit handoff. **STOP for independent Agent 1 re-audit** (PASS/REVISE/BLOCK).

## Remaining readiness blockers

B3 real checkpoint bytes / owner pin UNSET, production built-image runtime lock, tiny-Gemma revalidation, B2 F4/identities/double-BOS authorized evidence, hard cost cap and separate explicit scientific execution authorization. No new Q1 empirical outputs or protocol amendment approved.

**Owner precedence approval status: PENDING.** This document records the requested choice, not owner signoff.
