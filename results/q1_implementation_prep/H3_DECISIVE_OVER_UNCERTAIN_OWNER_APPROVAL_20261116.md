# TRACES Q1 — Research-Owner Approval: H3 Decisive-Over-Uncertain Precedence

**Decision recorded:** 2026-11-16 (America/Chicago), from the research owner's explicit **“yes”** in direct response to the narrowly stated approval question in the project conversation.  
**Scope:** documented H3-endpoint interpretation **APPROVED**; targeted non-forward engineering corrections permitted. This is a record of chat-expressed approval, **not** an independent signature or a formal protocol amendment.  
**Scientific authorization:** **OFF**. No pretrained/model loads, any model forwards (including tiny-Gemma), checkpoint downloads, GPU activity, cloud expenditure, empirical outcomes, change to frozen preregistration, or merge to main.

**Anchors:** frozen main `b1e8874f45311678e6fa241f504e7e09d4a7808c`; v1.3 SHA-256 `0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b`; second Agent 1 re-audit target code `3e44e950c2ba3ee6edc94b427c203e7f6cd8ccd1` (tree `9b78ba6d77890f43a672cbc94e63e7c3a1a67748`). Full prior request is `results/q1_implementation_prep/H3_AGENT1_SECOND_REAUDIT_REVISE_AND_OWNER_PRECEDENCE_REQUEST.md`; its PENDING status is historical and superseded by this signed-off conversational choice.

## Explicitly approved interpretation — H3 endpoint only

After model-level gates and applicable route are established, a **fully observed, definitive condition that already makes full H3 impossible** takes precedence over uncertainty in another, independent H3 or H1 measurement when classifying **that model's H3 endpoint**. Set `determinate_non_support`, and **also preserve and disclose the uncertain evidence**.

Concrete cases:
1. **R-moved**: both required 152-head signed upstream ranking maps are valid and complete; the unordered top-2 sets differ. Full H3 is disqualified, even with forward/reverse k* undetermined or H1-indeterminate. Record disagreement, both sets, k* uncertainty and the original H1 status. **Do not rewrite H1 as determinate.**
2. **R-moved**: reverse k* is definitively **>10** under frozen F2/interval rules, while forward k* is undetermined. Full H3 is disqualified; report both directional k* results.
3. With no definitively disqualifying condition, H1-indeterminate or required-direction k* in the undetermined region remains **H3-indeterminate** (subject to gate and applicability checks).
4. A missing or invalid ranking cannot establish definitive top-2 disagreement. Only valid complete forward **and** reverse 152-head maps can do so. Invalid measurements require their own technical incident handling; do not invent effects, conceal missingness, or promote invalid evidence to positive H3 support.
5. Failed §6 model-level integrity/validity gates retain `gate_failed` priority; do not override them with an H3 determinate result.
6. This precedence changes **only full-H3 endpoint status**, not the independent H1 endpoint, frozen route class, §5.4(3) two-indeterminate study rule, §5.4(4) 4/5 rule, H3a/H3b mutual exclusion, or any frozen threshold, membership, BOS choice or statistical method.

## Approved bounded engineering follow-up

Correct the exact Agent 1 second-reaudit findings, preserving already passed H3-A and per-model direction rules:
- H3-B: implement the above fully observed definitive-disqualifier precedence; reject unknown route labels like `R-weird` fail-closed rather than scientific determinate non-support; reject contradictory H3 endpoint records (e.g., `full_qualifying` with `full_h3_eligible=False` or invalid/missing pair) as technical-invalid, never canonical support.
- H3-C: require both exact fixed five primary model IDs `primary_l19_seed42` … `primary_l19_seed46` **and** `family == "l19"` for each; refuse canonical route and H3 claims on mismatch, retaining descriptive diagnostics. Names and declared family **do not prove real checkpoint provenance**; D5/B3 owner manifest pin remains unset.
- Integration: add non-forward end-to-end `run_study` tests reaching both H3a and H3b, plus adverse overlaps, unknown route, contradictory record and family spoofing. Run full permitted pinned CPython 3.11 CPU synthetic suite, record versions, results and F1 hash, and **STOP for independent Agent 1 re-audit**.

Previously approved BF16 F5, H3 applicable direction, H3-A mutual exclusion and H3-B uncertainty accounting stay in force. No model inference or execution authorization is granted.

**Other blockers remain:** B3 verified real bytes / owner pin, built-image environment lock, tiny-Gemma revalidation, B2 F4/identities/double-BOS and a separate explicit scientific execution and hard-budget authorization.
