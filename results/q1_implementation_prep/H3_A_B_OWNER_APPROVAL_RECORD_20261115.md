# TRACES Q1 — Research-Owner Approval of H3-A / H3-B Interpretations

**Decision date:** 2026-11-15 (America/Chicago)
**Decision authority:** Research owner, expressed explicitly in the project conversation as “Yes” in response to the question, “Do you approve these two clarifications, H3-A and H3-B?”
**Status:** **APPROVED AS DOCUMENTED INTERPRETATIONS**; code correction and independent Agent 1 re-audit still pending.
**Scope:** Approves only the interpretations below and corresponding engineering correction, including H3-C fixed-roster fail-closed defect repair. **Does NOT approve scientific/model execution**, a protocol amendment, checkpoint acquisition, cloud cost, or any change to frozen `main`.

## Anchors
- Frozen `main`: `b1e8874f45311678e6fa241f504e7e09d4a7808c`
- Frozen v1.3 protocol SHA256: `0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b`
- Previously approved F5 BF16 / H3 applicable direction: `F5_H3_OWNER_APPROVAL_INTERPRETATION_RECORD_20261115.md` (remains in effect)
- Agent 1 overall H3-B code verdict: **REVISE** on code `5fd62ff33f09b93b113996e778bd425f2bad50fb`, tree `4c28ec7937e32f232814a51c28a9965b5994985f`.
- Full proposed wording reviewed before approval: `H3_A_B_C_REVISE_OWNER_DECISION_AND_CORRECTION_PROPOSAL.md`, prepared at tip `3504bd5fb194251c36b3cd00520443aea295cae9`.
- This record is a faithful record of **chat-expressed owner approval**, not a fabricated signature or independently verified signer identity.

## H3-A — Owner-approved mutually exclusive study-level interpretation

Apply §5.3 and the frozen §5.4(4) **4-of-5** criterion only after individual qualification and endpoint inconclusive precedence.

1. With ≥4 full, applicable qualifying **fixed** primary seeds, if one unordered top-2 head pair occurs in ≥4 of the five, **H3a supported** and **H3b not supported**. Report any dissenting seed's pair and direction descriptively; 4+1 conservation is not a claim of 5/5 identity.
2. If ≥4 individually qualify and **no pair occurs in ≥4** of the five while qualifying pairs vary, **H3b supported**, **H3a not supported**.
3. Do not report H3a and H3b both supported. Do not turn missing, gate-failed or inapplicable seeds into invented pair differences. Preserve all prespecified thresholds and the complete signed directional data in reporting.
4. This is an owner-approved operational disambiguation, **not a protocol-text rewrite**.

## H3-B — Owner-approved H3-endpoint uncertainty / §5.4(3) interpretation

For each of the fixed primary seeds, classify H3 endpoint evidence without replacing, silently dropping or cherry-picking models:

- **Fully qualifying:** valid model gates, applicable route and denominator, explicit H1-fail, complete 152-upstream-head directional ranking, frozen cross-fit `k* ≤ 2` in every applicable direction, and (R-moved only) matching unordered top-2 across both directions.
- **H3-indeterminate:** an applicable route with H1-indeterminate, undetermined `k*` in a required direction, or missing/incomplete required H3 measurements; distinguish technical missingness from valid statistical uncertainty.
- **Determinately non-supporting / inapplicable:** R-tag or R-intermediate, explicit H1-pass, definitive `k* > 10`, or complete R-moved forward/reverse top-2 disagreement. A failed applicable direction-denominator gate makes that direction inapplicable under frozen §5.3; report it explicitly. A route classified `Indeterminate` is itself indeterminate (not determinate non-support).
- **Gate-failed:** failed §6 model validity/integrity gate, reported separately with no replacement.

**H3 study-level precedence:** If the route-level study outcome is INCONCLUSIVE **or** at least 2 of the fixed five are `gate-failed` or `H3-indeterminate`, the **H3 endpoint is INCONCLUSIVE**, and neither H3a nor H3b can be supported. With exactly one H3-indeterminate and four full qualifying seeds, 4/5 H3a may still be supported. Inapplicable/non-supporting seeds cannot be silently reclassified as uncertainty or vice versa.

## H3-C — Approved engineering-only integrity correction

Enforce exactly the frozen five primary identities `primary_l19_seed42`, `primary_l19_seed43`, `primary_l19_seed44`, `primary_l19_seed45`, `primary_l19_seed46`, with no duplicates/substitution, for **both** the canonical route-level decision and H3 study aggregation. Refuse canonical study support for any roster mismatch even if five other IDs are present.

## Implementation and review boundary

Cursor may implement only H3-A/B/C in directly affected analysis/driver code and non-forward synthetic tests, then report code SHA/tree and run the full pinned CPU suite. **STOP for independent Agent 1 PASS/REVISE/BLOCK** before engineering closure. Accepted per-seed H3-B direction logic, B1/S2/D5 provenance, F1/F2, frozen 50/200 allocation, single-BOS and other preregistered rules remain unchanged.

**Scientific authorization remains OFF.** B3 real checkpoint verification/owner pin, built-image runtime lock, B2/F4/identities, tiny-Gemma authorization, hard budget and separate scientific execution approval remain pending.
