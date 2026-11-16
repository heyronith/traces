# TRACES Q1 — Cursor Narrow Remediation Handoff: H3-A / H3-B / H3-C (Owner Approved)

**TASK:** Correct exactly the three independent Agent 1 H3 study-aggregation findings and add adversarial **synthetic/non-forward CPU tests**. **STOP after publication for Agent 1 independent re-audit.**

**Branch:** `q1-impl-audit` (do not touch frozen `main`).
**Correction baseline code:** `5fd62ff33f09b93b113996e778bd425f2bad50fb` / tree `4c28ec7937e32f232814a51c28a9965b5994985f`.
**Evidence tip after Agent 1 H3-B review:** `1284eac4e1fd2354ca06929bb4f2cf21c926a512`.
**Owner-approved interpretation:** `results/q1_implementation_prep/H3_A_B_OWNER_APPROVAL_RECORD_20261115.md` (commit `4bfde5e17cec187c99895bd75dc4d2af37dc3496`).
**Original issue analysis:** `H3_A_B_C_REVISE_OWNER_DECISION_AND_CORRECTION_PROPOSAL.md`.
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c`.
**Scientific authorization:** **OFF** throughout. No pretrained downloads/loads, GPU, model forwards including tiny-Gemma, cloud expenditure, empirical outcomes, production pins, protocol edits or merge.

## Existing independent verdict; preserve passed areas

Agent 1 returned **H3 engineering REVISE**, with per-model R-and forward / R-or reverse / R-moved two-direction agreement and k*, explicit H1-fail, gates, 152 upstream head checks, F1/F2 and accepted B1/S2/D5 **PASS**. Reopen only with a new reproducible regression. Agent 1 reproduced focused synthetic cases and F1; did **not** rerun the pinned CPython 3.11 full suite. Cursor's prior full-suite **351 passed, 12 skipped** is only a reported baseline, not a new success claim.

## Fix H3-A — Mutually exclusive H3a / H3b

Location: `src/traces/q1/analysis_ext.py` near `h3_study_level_from_seed_records` support logic.

- Under fixed roster and §5.4 precedence, **H3a supported** when **≥4 individual seeds qualify** and the **same unordered pair is present in ≥4 of the five**. When H3a holds, H3b MUST NOT be marked supported even if a fifth qualifying model differs; disclose the minority pair descriptively.
- **H3b supported** when ≥4 seeds individually qualify, qualifying top-2 sets vary, **and no unordered pair is conserved in ≥4/5**.
- Never label both H3a and H3b as supported. Preserve full five-seed pair variation information without promoting a descriptive 4+1 difference into H3b.
- Update the prior test `test_h3a_conserved_and_h3b_varied_both_reported` to explicitly reject dual support; add 5/5, 4+1, 3+2, and 4 eligible/one inapplicable cases.

## Fix H3-B — H3-specific endpoint uncertainty & §5.4(3)

Locations: `src/traces/q1/driver.py` H3 study aggregation and `src/traces/q1/analysis_ext.py` per-seed H3 evidence/state.

- Preserve **route-level** study label separately; derive an independently audited **H3 endpoint status** for every frozen primary seed.
- Explicitly classify every seed as:
  - `full_qualifying`: complete applicable H3 evidence, H1-fail, valid route-denominator(s), frozen F2 k*≤2, and for R-moved both directions' unordered top-2 match.
  - `h3_indeterminate`: within applicable route, H1-indeterminate, a required k* in the undetermined band, or required H3 measurement missing/incomplete. Record exact reason and whether numerical uncertainty or technical evidence missing. Treat route-class `Indeterminate` as indeterminate.
  - `determinate_non_support` / `inapplicable`: valid R-tag/R-intermediate, explicit H1-pass, definitive k*>10, fully measured R-moved top-2 disagreement, and directional route-denominator inapplicability per frozen §5.3; keep reason-specific distinctions.
  - `gate_failed`: a §6 gate failed; keep its identity and report, never replace it.
- Apply §5.4(3) to the **H3-relevant endpoint**: `route_study_INCONCLUSIVE OR (number of gate_failed + h3_indeterminate among exact five >= 2)` -> **H3 study-level INCONCLUSIVE**, with no H3a/H3b supported.
- If one H3-indeterminate plus four qualifying with same pair, H3a may still support (if route study not INCONCLUSIVE). Three qualifying with two determinate non-support => **not supported**, not necessarily INCONCLUSIVE. Two H1-indeterminate or two undetermined k* => INCONCLUSIVE. Don't make a fixed four-seed subset by cherry-picking.
- Keep per-seed and study-level evidence/labels distinct; do not silently suppress technical-failure incidents.

## Fix H3-C — Exact frozen five-model identities

Location: `src/traces/q1/driver.py` code currently falling back to `roster_ids` for five unrecognized IDs.

- Require the **exact** unordered set `primary_l19_seed42` through `primary_l19_seed46`, no duplicates, for **both** canonical route-level `study_level_decision` and H3 aggregation. Derive expected set from frozen `REQUIRED_PRIMARY_SEEDS`. Never replace expected with the observed list on mismatch.
- Missing, renamed, repeated or substituted model -> explicit fail-closed canonical study result, including mismatch details. Per-unit descriptive records may be retained but **not** mislabeled as frozen cohort support. Consider whether current earlier duplicate checks silently collapse dicts.
- Valid five with one inapplicable seed must keep non-supporting identity without erasing four valid seeds.

## Regression boundaries / tests

- Preserve per-model applicable-direction R-and/R-or/R-moved behavior, H1-fail in ≥1 direction for R-moved under frozen §5.2, and both-direction k*≤2 plus matching sets for full R-moved H3. No forward-only fallback.
- Preserve signed ranking, exact 152 upstream head population, tie-break `(layer, head)`; no downstream-56 or core-triple leakage; frozen `F2` fold positions and `F1` stream hash `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`.
- Preserve the other accepted B1/R1/M1/D2–D4/M2/S2/D5 trust/auth contracts; F5 `torch.bfloat16` + `autocast_adapter_dtype=False`; single BOS; frozen 50/200 IDs; experimental roles; exploratory stop; scientific threshold values and protocol bytes.
- New/updated tests must include: H3a 4+1 but H3b absent; H3b with 3+2 among five qualifying; 4 same + 1 inapplicable still H3a; 1 H3-indeterminate + 4 H3a supports; 2 H1-indeterminate => H3 INCONCLUSIVE; 2 k* undetermined => H3 INCONCLUSIVE; 2 gate failures => H3 INCONCLUSIVE; 2 determinate inapplicable => no false INCONCLUSIVE; R-moved fully measured disagreement => determinate non-support; route INCONCLUSIVE blocks H3; five arbitrary IDs / one renamed / duplicate / missing IDs refuse route AND H3 study support; valid fixed IDs accept.
- Check that testing does not pass purely via hard-coded fixture states divorced from the real `run_study` orchestration. Include mock end-to-end driver paths where possible **without any forward**.

## Required deliverables

1. Minimal corrective code and tests committed to `q1-impl-audit`; report **full correction commit SHA, tree SHA, clean tested tip, published tip**, parents and exact changed files.
2. Run **full permitted pinned CPython 3.11.15 CPU suite** (same scope as previous 351 passed/12 skipped baseline), plus focused new adversarial tests; archive real command, versions, full pass/fail/skip output. No model forward.
3. Independently recheck F1 SHA256; leave `PINNED_CHECKPOINT_MANIFEST_SHA256=None`; do not create real checkpoint manifests.
4. Archive correction matrix, synthetic CPU test record, and narrowly scoped **Agent 1 re-audit handoff** in `results/q1_implementation_prep/`. Ensure evidence-only publication after code commit changes no code.
5. **STOP for Agent 1**, requesting independent per-finding PASS/REVISE/BLOCK for H3-A/B/C, exact roster and §5.4 precedence; preserve previously accepted per-model H3-B and B1/S2/D5 scopes.
6. Do not claim scientific readiness. B3 byte provenance and owner pin, real runtime lock, B2 empirical/F4/identities, tiny-Gemma authorization, budget and separate scientific execution remain blocked.

**No research-owner decisions remain pending for H3-A/H3-B's specifically recorded interpretations. Any newly identified ambiguity requires a new owner decision before applying unapproved scientific rules.**
