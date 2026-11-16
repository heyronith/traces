# TRACES Q1 — Research-Owner F5 / H3 Interpretation Decision Record

**Decision date (conversation):** 2026-11-15 (America/Chicago)  
**Status:** RESEARCH-OWNER DECISIONS EXPLICITLY COMMUNICATED IN CONVERSATION; H3 IMPLEMENTATION AND INDEPENDENT RE-AUDIT PENDING. This file is a record of stated approvals, not a cryptographic signature, a protocol amendment, or permission for any model forward.  
**Scientific authorization:** OFF. No pretrained loads, model forwards, checkpoint downloads, GPU activity, cloud spend, empirical analysis, or merge to frozen `main` authorized.  
**Frozen protocol:** v1.3 SHA256 `0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b`; frozen `main` `b1e8874f45311678e6fa241f504e7e09d4a7808c`.  
**Accepted B1 engineering:** Agent 1 independent PASS on code `3730e892ab3f4f839570b8b4d8710a63f86efc52`, tree `6a133d8e3a52fe7bd3c652ec1770ac419665c6d1`.

## Owner-approved F5 interpretation: uniform BF16

The research owner explicitly approved **F5 BF16** after Agent 1's recommendation. Use `torch.bfloat16` for the base and the LoRA adapter, with `autocast_adapter_dtype=False`. On load, stored FP32 adapter parameters are cast/rounded to BF16 once for Q1 inference and must not silently upcast or fall back. Verify the post-load dtype contract; `attn_implementation="eager"` and the frozen runtime pins are unchanged.

This matches the current proposed implementation and is a documented interpretation of §16's model BF16 policy, **not** new scientific execution authorization. The interpretive record must disclose that rounding FP32 stored adapter weights once need not be bitwise identical to mixed-precision training with FP32 master weights, and that Stage 1 execution numerics are not proven directly comparable. F4 calibration is performed only **after separate forward authorization** and under this fixed F5 policy; no F4 evidence is claimed here.

## Owner-approved H3 interpretation: applicable-direction top-2

After Agent 1's explanation, the research owner explicitly approved the applicable-direction rule. For the **five primary l19 seeds**, first assign route class from the frozen §5.1 confidence-interval gates and establish relevant route-denominator validity; **do not choose a direction to maximize H3**.

- **R-and:** use the forward signed upstream single-head ranking.
- **R-or:** use the reverse signed upstream single-head ranking.
- **R-moved:** both directions must individually satisfy the applicable direction criteria, including the frozen `k* ≤ 2` criterion, and yield the **same unordered top-2 head set** under the frozen tie-break. Disagreement is reported as directional/partial only and the seed counts as **non-supporting** of a full H3 hypothesis; do not use forward-only or a favorable union.
- **R-tag, R-intermediate, indeterminate, failed route-denominator direction or model-level gates:** no full H3 support; report descriptive effects and mark inapplicable/gate-failed.
- **H1-fail:** must be a genuine, explicit H1-fail rather than H1-indeterminate. Under R-moved, H1-fail in at least one required direction is the frozen eligibility condition; keep directional H1 observations separately.
- **Across seeds:** use the §5.4(4) **at least 4 of 5** full, applicable, individually qualifying seeds rule for H3a/H3b study-level support; inapplicable seeds count as non-supporting. H3a uses a conserved identical unordered pair across qualifying seeds; H3b uses varying top-2 sets among qualifying seeds. Disclose each seed's route class, direction(s), identity, and non-supporting reason. Preserve the frozen study-level indeterminate precedence and no pooled p-values.

**Interpretation basis:** frozen v1.3 §§5.1–5.4, especially §5.4(2) 'direction-appropriate head hypotheses', §5.4(4) ≥4/5 and the §5.2 R-moved full-bidirection convention. No hypothesis threshold, data allocation, BOS rule, stop rule, or confirmatory condition count is changed.

## Engineering follow-up and clearance

1. **Cursor:** implement the H3 interpretation in `src/traces/q1/driver.py` and directly affected analysis plumbing only, with focused negative and positive non-forward synthetic tests for R-and/R-or/R-moved, denominator gates, H1-fail vs indeterminate, tie-breaking, 4/5 H3a/H3b and no selective fallback. Do not change accepted B1/S2 or frozen protocol files.
2. **Agent 1:** independently verify implementation against this written interpretation and frozen §5 criteria; record exact tested commit and tree, tests and PASS/REVISE/BLOCK.
3. **Owner:** approve any formal signature or additional provenance format required by the project's sign-off policy. This document records chat-expressed approval but does **not** impersonate a handwritten/digital signature.
4. **Execution gate:** remains **NOT READY** until real B3 byte verification and owner pin, the real built-image runtime lock, authorized tiny-Gemma / F4 / identities / double-BOS evidence (B2), budget/throughput approval, and **separate explicit scientific execution authorization**.

**Reference decision requests (archival, superseded as to selected interpretation):** `results/q1_implementation_prep/F5_OWNER_DECISION_NOTE.md`, `H3_DIRECTION_OWNER_DECISION_REQUEST.md`, `OWNER_DECISION_MEMO_F5_H3.md`. These earlier documents' 'pending' statements describe the state **before** this conversational approval; do not treat them as current owner status.

**Real-model execution remains prohibited.**
