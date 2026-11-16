# TRACES Q1 — Agent 1 H3-A/B/C REVISE: Owner Interpretation and Narrow Correction Proposal

**Status:** PROPOSAL ONLY; H3-A and H3-B OWNER APPROVAL PENDING. No implementation authorization for these two interpretive changes is inferred from the earlier H3-B applicable-direction approval.
**Baseline audited code:** `5fd62ff33f09b93b113996e778bd425f2bad50fb`; tree `4c28ec7937e32f232814a51c28a9965b5994985f`; published tip at independent re-audit `1284eac4e1fd2354ca06929bb4f2cf21c926a512`.
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`; v1.3 SHA256 `0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b`.
**Prior owner approval:** `results/q1_implementation_prep/F5_H3_OWNER_APPROVAL_INTERPRETATION_RECORD_20261115.md` (F5 uniform BF16 and H3 per-seed applicable direction). This approval remains valid; H3-A/B are narrower NEW interpretations requiring separate explicit owner consent.
**Scientific authorization:** OFF. No pretrained loads, checkpoints downloads, model forwards, GPU work, spending, empirical results, frozen-protocol changes, or merge to main permitted.

## Independent Agent 1 result (as reported in review)

**Overall H3 engineering: REVISE** (not PASS). Per-seed R-and/R-or/R-moved direction selection, explicit H1-fail gates and F1/F2 preservation: **PASS**. Three aggregation findings:
- **H3-A (major)** `src/traces/q1/analysis_ext.py` H3a/H3b dual support when four qualifying seeds share an unordered pair and fifth differs; prior synthetic test explicitly expected dual support.
- **H3-B (major)** `driver.py` routes only route-level INCONCLUSIVE into H3; H3-specific uncertainty (e.g. H1-indeterminate or undetermined concentration in two seeds) does not trigger §5.4(3) for H3.
- **H3-C (minor–major)** `driver.py` falls back to arbitrary observed five model IDs instead of the exact frozen primary seed roster, permitting unapproved substitutions.

Agent 1 independently reproduced focused probes and F1 hash, but **did not independently run the full pinned CPython 3.11.15 351-test suite**. The 351 passed / 12 skipped count remains **Cursor-reported** only.

## Proposed owner interpretation 1 — H3-A: mutually exclusive full-study labels (PENDING)

Frozen §5.3 defines H3a as the *same unordered top-2 pair in at least four of five* and H3b as *seed-varying top-two*. Literal variation can coexist with 4/5 conservation; the frozen text does not expressly state an exclusive priority. The earlier handoff explicitly requested transparent dual reporting, so engineering must **not** silently invert the test without owner approval.

**Recommendation for owner approval:**
- If at least 4/5 fixed primary seeds individually qualify and the same unordered pair occurs among **at least four** of the five, report full **H3a supported; H3b not supported**.
- If ≥4/5 individually qualify, **no pair is shared by ≥4**, and the qualifying sets differ, report full **H3b supported; H3a not supported**.
- Continue to disclose all five seed-level head pairs and any exceptions. A 4+1 pattern is **conserved under the preregistered 4/5 criterion**, not wholly identical; report the dissenting seed descriptively.
- If neither condition holds, do not assert H3a or H3b; preserve inconclusive and other gates.

This is a **documented interpretation proposal**, not a change to the frozen thresholds. Owner may approve, reject, or request another reviewed rule **before experimental outcomes**.

## Proposed owner interpretation 2 — H3-B: H3-endpoint inconclusive accounting (PENDING)

Frozen §5.4(3) states that ≥2/5 gate-failed or *indeterminate for the relevant endpoint* produces study-level **INCONCLUSIVE**. §5.3 distinguishes confirmed `k*≤2`, confirmed `k*>10` and an undetermined concentration zone.

**Recommendation for owner approval:** Give each fixed primary model an auditable, non-overlapping H3 endpoint state:

1. **Qualifying/full:** passes model gates and applicable denominator(s), has explicit H1-fail, complete 152-head signed rankings, `k*≤2` under frozen F2 in every required direction and (for R-moved) identical forward/reverse unordered top-2.
2. **H3-indeterminate:** otherwise applicable route with H1-indeterminate; an undetermined `k*` zone in any otherwise required direction; or missing/incomplete required H3 evidence. Preserve the precise reason; distinguish a scientific indeterminate result from a technical failure requiring investigation.
3. **Determinately non-supporting/inapplicable:** valid R-tag or R-intermediate class, explicit H1-pass, definitively `k*>10` with valid directional evidence, or complete R-moved evidence with different directional pairs. A direction failing the route-denominator gate is **inapplicable** under frozen §5.3; record it separately rather than pretending `k*≤2`. Route-class `Indeterminate` is treated as indeterminate, not a determinate R-tag/R-intermediate result.
4. **Gate-failed:** a §6 model validity/integrity gate failed; do not replace the model or call it qualifying.

**Precedence:**
- If route-level study conclusion is INCONCLUSIVE **or** `count(gate-failed + H3-indeterminate) >= 2`, H3 **study-level INCONCLUSIVE**; no full H3a/H3b claim, even if other seeds have apparent conserved pairs.
- If exactly one H3-indeterminate and four fully qualifying seeds meet H3a, H3a may still be supported, subject to the other frozen requirements and H3-A owner decision.
- Distinguish INCONCLUSIVE from determinate non-support; do not silently select a favorable four-seed subset.
- Report route-level and H3-endpoint conclusions separately.

Question for owner: Approve this H3-indeterminate/definitely-non-supporting breakdown as the operational meaning of frozen §5.4(3)?

## H3-C — Engineering fail-closed repair (no new scientific choice)

- Require exact unordered identity set `{primary_l19_seed42,...,primary_l19_seed46}` **with no duplicates or substitutions** for both the canonical route-level study decision and H3 study aggregation; use frozen `REQUIRED_PRIMARY_SEEDS` only.
- Missing, renamed, repeated or substituted primary IDs -> explicitly refuse canonical study claims, record roster mismatch and missing/unexpected identities; do not fall back to observed identities even when there are exactly five.
- Preserve eligibility/inapplicability within the **correct** fixed roster; failing a model's gate is not permission to drop or replace its identity.

## Cursor correction handoff — ONLY AFTER owner consents to H3-A/B

1. Restrict edits to `src/traces/q1/analysis_ext.py`, `src/traces/q1/driver.py`, and focused synthetic tests, plus resulting test/evidence reports. Avoid revisiting accepted per-model H3-B direction logic, B1, S2, D5 or F5. No preregistration changes.
2. Implement H3-A single full-study label under 4/5 conserved-pair precedence, explicitly revise `test_h3a_conserved_and_h3b_varied_both_reported` to prohibit dual full support for 4+1. Show dissenting pair.
3. Implement H3-B H3-specific endpoint states and §5.4(3) precedence, including two H1-indeterminate, two undetermined k*, one indeterminate plus four conserved, R-moved disagreement, missing measurements, and ≥2 gate failures. Do not conflate R-intermediate, R-tag and route-Indeterminate.
4. Implement H3-C fixed roster validation before any primary canonical study label; test five arbitrary IDs, single renamed/substituted model, duplicate unit ID, missing unit and correctly fixed five with one inapplicable seed.
5. Assert 4/5 eligible/1 inapplicable H3a; 3/5 eligible/2 determinate non-support does not become H3 supported; H3b only when no qualifying pair is 4/5 conserved. Preserve complete 152-head signed map, F2 fold positions, k* interval rules, no downstream leakage, F1 fixed stream, frozen 50/200 and descriptive roles.
6. Run pinned CPython 3.11.15 full permitted CPU synthetic suite on clean committed tree and record exact counts/versions; log F1 SHA256 `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`. No model forwards, including tiny-Gemma.
7. Publish minimal code commit/tree, CPU test record, correction matrix, and targeted Agent 1 re-audit handoff; **STOP** until independent Agent 1 PASS. Distinguish Cursor-reported vs independently reproduced results.

**Open gates remain unchanged:** B3 real provenance and owner-controlled pin UNSET, actual built-image runtime lock, B2 empirical evidence/F4/identities, tiny-Gemma authorization, budget review and distinct scientific execution authorization.

## Owner approval box — INTENTIONALLY UNSIGNED

- H3-A mutually exclusive H3a / H3b 4/5 precedence: **PENDING**
- H3-B H3-specific indeterminate classification / §5.4(3): **PENDING**
- H3-C fixed-roster fail-closed engineering repair: **READY FOR IMPLEMENTATION**
- Owner expression of approval, date and source: **NOT YET PROVIDED**
- Scientific execution authorization: **OFF**
