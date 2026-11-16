# TRACES Q1 — Cursor H3 Direction Interpretation: Targeted Implementation Handoff

**Status:** OWNER-APPROVED INTERPRETATION / CODE CHANGE NOT YET PERFORMED  
**Branch:** `q1-impl-audit`  
**Baseline tip at preparation:** `a0931e58b0e84801fec7ce12b269ac23724370a4`  
**Frozen main:** `b1e8874f45311678e6fa241f504e7e09d4a7808c` — do not modify or merge.  
**Owner decision record:** `results/q1_implementation_prep/F5_H3_OWNER_APPROVAL_INTERPRETATION_RECORD_20261115.md`  
**Scientific authorization:** OFF; **no model forwards or downloads**.

## Task

Implement **only** the owner's approved H3 applicable-direction interpretation. Do not amend frozen Q1 v1.3 scientific files or change thresholds, seed allocation, F2 folds, H1/H2 logic, exploratory stop, BOS, workload catalog or existing B1/S2 provenance and evidence guards. F5 is owner-approved BF16, and already implemented; no F5 loader change is requested.

## Confirmed baseline issue

`src/traces/q1/driver.py` around lines 920–934 computes `top2_by_seed` from `u.head_effects_fwd` for **all** five primary models, before evaluating each unit's route class, gates, route denominators, H1 state and directional k*. The same map is then supplied to each `run_model_unit_analysis`.

`analysis_ext.py:h3_top2_conservation` currently requires **exactly five valid top-2 sets**, although frozen §5.4(4) supports a full head-level study claim when **at least 4 of 5** primary models qualify; an inapplicable fifth is non-supporting, not automatically a missing data field. `analysis.py:evaluate_h2_h3` can label H3b based on variation across even nonqualifying seeds. Resolve these as one targeted H3 correction, preserving fail-closed behavior elsewhere.

## Owner-approved H3 semantics — exact

1. Apply each model's frozen §6 validity gates, §5.1 route class and §4 direction-specific route-denominator gate **before** H3 direction selection.
2. `R-and`: signed upstream single-head **forward** ranking only.
3. `R-or`: signed upstream single-head **reverse** ranking only.
4. `R-moved`: assess **both** directions independently. Full H3 seed eligibility requires valid denominators, complete directional upstream ranking, **k* ≤ 2 in each direction**, and **identical unordered top-2 sets** in forward/reverse. If sets disagree, record both and classify as directional/partial, **not full seed H3 support**. A favorable direction cannot substitute for the other.
5. `R-tag`, `R-intermediate`, `Indeterminate`, gate-failed or direction-denominator failure: no full H3 support. Describe available signed effects; never supply a synthetic default pair.
6. Full H3 support also requires **explicit H1-fail**, not `H1-indeterminate`. For `R-moved`, preserve frozen H1-fail if **at least one** applicable direction decisively fails; do not alter the H1 rule.
7. Preserve frozen **k* ≤ 2** from F2 out-of-fold joint effects and its 90% lower bootstrap requirement, not a two-largest-head point estimate as a proxy.
8. Compare the qualifying models' unordered signed-effect top-2 head sets with frozen tie-break by `(layer, head)`; keep maps restricted to all **152 upstream** heads (never mix in the 56 downstream heads).
9. For study-level H3a/H3b, apply §5.4(4): **at least 4 of 5** primary models must **individually qualify** for full, applicable H3 consideration. Inapplicable/nonqualifying models count as non-supporting; disclose their reasons. H3a requires a common unordered pair in ≥4 of 5 qualifying seeds. H3b evaluates variation of the individually qualifying sets, not variation from nonqualifying models. Do not introduce an unapproved tie-break, H3a/H3b precedence or new threshold if their criteria overlap; show both conditions transparently for independent methodology review.
10. Preserve §5.4(3): if at least two models are gate-failed or indeterminate for the relevant endpoint, the **study-level** outcome is INCONCLUSIVE. Do not label a full H3 claim by skipping this precedence.
11. No across-seed p-values. Report each seed's route class, applicable direction(s), forward/reverse top-2 if applicable, gate/denominator status, complete ranking status, H1 state, directional k* classification, full-H3 eligibility and reason if non-supporting.
12. The primary H3 interpretation applies to five `l19` seeds 42–46 only. Do not change the three `l1523` exploratory seeds, the two secondary replication seeds, adapter-off reference or the scientific catalog.

## Implementation requirements

- Refactor the primary analysis so that top-2 selection depends on each unit's **computed gate/route/H1/F2** evidence. Avoid using a forward-only pre-analysis group map.
- Ensure group H3a/H3b uses only **eligible, complete** top-2 sets and explicit five-seed roster membership, while permitting a missing/inapplicable fifth to count non-supporting. Never select the best 4 of 5 based on results; evaluate the fixed five identities.
- Keep per-unit scientific claims and study-level claims distinct; don't infer a supported study-level H3 merely from one seed's H3 result.
- Preserve all previously accepted B1, R1, M1, D2–D4, M2, S2 and D5 independent authentication. Do not modify the F1 bootstrap stream, exact 50/200 splits or protocol files.
- Preserve current serialization/output contracts where possible. Add explicit auditable H3 direction and qualifying-set metadata; missing/indeterminate values must be explicit, not forged defaults.
- If a genuine new ambiguity is uncovered, **stop with an exact question and code location** rather than invent a scientific rule.

## Required negative + positive synthetic tests (no forwards)

Create focused new tests, minimally update old tests to reflect the approved rule, then run the **entire permitted pinned CPU suite** on a clean committed tree.

- All five primary `R-and` → forward top-2.
- All five primary `R-or` → reverse top-2 even when forward map would yield a different pair.
- Mixed `R-and` / `R-or` five-seed cohort uses each seed's applicable direction; no favorable-direction selection.
- `R-moved` agreeing forward/reverse top-2 and both `k*≤2` → potentially eligible if H1-fail and gates pass.
- `R-moved` disagreeing top-2, one direction failing `k*≤2`, incomplete 152-head map or failed denominator → non-supporting, both directions reported.
- `H1-indeterminate` or `H1-pass` → never H3 support even if heads coincide.
- A fifth inapplicable seed does **not** erase four complete qualifying seeds, and cannot create spurious H3b variation.
- Three qualifying seeds + two inapplicable/nonqualifying → no full H3 claim.
- Frozen H3a 4-of-5 conserved pair; H3b varied qualifying sets; explicit check for simultaneous-criteria behavior without invented precedence.
- 2/5 gate-failed or indeterminate → study-level INCONCLUSIVE precedence.
- Signed ranking and deterministic tie-break; 56 downstream heads never alter upstream top-2.
- Full primary/exploratory/secondary/adapter-off mock-driver path preserves F1/F2, no forward-only H3 fallback, integrity evidence and existing conditions.
- Verify F1 reference SHA256 `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`; previous independent permitted suite `333 passed, 12 skipped` is **baseline only**, not a target that substitutes for rerunning.

## Delivery / independent audit

1. Commit the minimum code and synthetic tests on `q1-impl-audit`. Report exact correction commit SHA and **tree SHA**, baseline, tip and frozen-main SHA.
2. Archive a focused H3 correction matrix, pinned-environment complete CPU test output, and Agent 1 targeted re-audit handoff under `results/q1_implementation_prep/`.
3. Explicitly disclose new/changed tests, any failures/skips and the exact preserved F1 hash; **do not** claim independent PASS.
4. **STOP for Agent 1** to independently inspect and reproduce the H3 change. Ask PASS/REVISE/BLOCK for: per-seed direction, R-moved bidirection and k*, correct four-of-five full seed qualification, H3a/H3b aggregation and study inconclusive precedence; regressions against accepted B1/S2.
5. Leave B3 real-byte provenance/owner pin, built-image lock, tiny-Gemma revalidation, B2 F4/identities/double-BOS, hard budget and separate scientific run authorization as **NOT READY**.

**Not permitted:** pretrained loads/downloads, GPU, model forward of any size, cloud expenditure, new empirical outcomes, protocol amendments, modifying frozen `main`, setting `PINNED_CHECKPOINT_MANIFEST_SHA256` or enabling scientific authorization.
