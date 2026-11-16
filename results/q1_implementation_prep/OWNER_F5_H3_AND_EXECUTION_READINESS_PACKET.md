# TRACES Q1 — Research-Owner Decisions and Scientific-Readiness Packet

**Purpose:** Request decisions; do **not** presume approvals or change frozen scientific text.  
**Status:** OWNER DECISIONS PENDING / scientific authorization OFF.  
**Frozen protocol:** v1.3 SHA-256 `0a88d85bd4d94412f2568389dc16bdf51f7008cd01400e3e7ea0ab79de0d529b` at frozen `main` `b1e8874f45311678e6fa241f504e7e09d4a7808c`; direct pins v1.2 SHA-256 `4a90dd7133719a751400c456db615e726a245d9a34073bf2a212048`.  
**Accepted B1 code/tree:** `3730e892ab3f4f839570b8b4d8710a63f86efc52` / `6a133d8e3a52fe7bd3c652ec1770ac419665c6d1`, Agent 1 PASS at `results/q1_implementation_prep/B1_D5_A1_B1_A2_AGENT1_REAUDIT_VERDICT.md`.  
**Engineering evidence:** independent permitted suite 333 passed, 12 skipped; F1 stream SHA-256 `fe4641876e74ed5d70fde3e20ff1f30be22b6ac606c02553abfb6005ac6790ed`. Neither is real-checkpoint validation.

## Decision 1 — F5 adapter numerical policy (owner; REQUIRED)

Existing source: `F5_OWNER_DECISION_NOTE.md`. Choose **one** or explicitly defer:

- [ ] **F5-A:** BF16 base + BF16 LoRA; `autocast_adapter_dtype=False` (currently implemented as proposed engineering behavior; **not approved by its implementation**).
- [ ] **F5-B:** BF16 base with FP32 adapter policy; specify exact PEFT flag, casting semantics, post-load audit change and comparison to earlier runs. Assess scientific/protocol implications before any code change.
- [ ] **F5-DEFER:** numerical comparison or methods review required; specify evidence and authorization needed.

**Required written explanation:** precision choice, prior-run comparability, reproducibility and tolerance/identity impact. Select **before** confirmatory model forwards. Do not silently choose A because code currently uses it.

Owner selection: **PENDING**  
Scientific rationale / exact flags: **PENDING**  
Owner signer / UTC date / approved artifact SHA: **PENDING**

## Decision 2 — H3 top-two conservation direction (owner; REQUIRED)

Existing source: `H3_DIRECTION_OWNER_DECISION_REQUEST.md`. Choose **one** or explicitly defer:

- [ ] **H3-A:** Forward-only head effects for all five primary seeds (current engineering behavior).
- [ ] **H3-B:** Applicable-direction head effects: forward for R-and, reverse for R-or; **owner must specify exact R-moved aggregation** (e.g. both/union/intersection), seed aggregation and tie-breaking where needed. Do not let engineering invent missing rules.
- [ ] **H3-C:** Other precisely specified rule with citation to frozen v1.3 and rationale.
- [ ] **H3-DEFER:** methodology review required; leave current behavior flagged and scientific execution blocked.

**Required written explanation:** relationship to frozen hypothesis, applicable direction by route class, R-moved handling and prespecified threshold interpretation. If selection would alter the preregistered analysis, route through separate formal owner/methods review; do not silently amend v1.3.

Owner selection: **PENDING**  
Explicit R-moved rule / rationale / citation: **PENDING**  
Owner signer / UTC date / approved artifact SHA: **PENDING**

## Readiness gates — separate from the two decisions

| Gate | Current state | Evidence needed to clear |
|---|---|---|
| **B1 static engineering** | **PASS — CLOSED** | Agent 1 accepted code/tree and full permitted synthetic suite; no reopening without a new confirmed defect |
| **F5, H3** | **BLOCKED** | Explicit research-owner decisions and reviewed implementation/interpretation |
| **B3 checkpoint identity** | **BLOCKED** | Actual verified real base / ten adapter / tokenizer/template file bytes; independent audit; owner-controlled canonical manifest and `PINNED_CHECKPOINT_MANIFEST_SHA256` |
| **Production runtime lock** | **BLOCKED** | Capture and independently verify CPython, transitive wheel SHA-256, base image, CUDA/driver/GPU hardware and determinism from the actual intended built image; synthetic fixture is not a lock |
| **B2 empirical integrity** | **BLOCKED** | Separately authorized real-model F4 numerical repeats, identities 1–9, double-BOS descriptive scan and frozen gate evidence; preserve S2 provenance requirements |
| **Tiny-Gemma revalidation** | **AUTHORIZATION PENDING** | Narrow written permission for CPU-only randomly initialized suite on the exact current committed tree; prior 16/16 result does not cover changed source |
| **Scientific execution** | **OFF** | Distinct written research-owner approval after all applicable gates and budget/throughput review; no inference merely because a checklist was prepared |

## Approval / authorization requests (do not combine implicitly)

1. **Scientific-numerics interpretation**: F5 decision (above).
2. **H3 interpretation**: choose precise applicable-direction rule (above).
3. **B3 byte acquisition/inspection**: if desired, separately define permitted repositories/revisions, read-only versus download scope, environment, budget, duration and artifact retention. This document itself authorizes **none** of these.
4. **B3 manifest owner freeze**: after real bytes and independent audit, approve exact canonical digest and evidence. Leave `PINNED_CHECKPOINT_MANIFEST_SHA256=None` until then.
5. **Runtime built-image verification**: approve construction and verification separately before incurring cloud/GPU costs.
6. **Tiny-Gemma**: separately authorize `TRACES_Q1_ALLOW_TINY_GEMMA_FORWARD=1` for the exact committed tree and CPU-only random fixture if needed.
7. **B2 empirical and final scientific execution**: separate signed approvals with scope, budget, integrity conditions and stop criteria. No blanket approval here.

**Reference:** `results/q1_implementation_prep/B3_CHECKPOINT_PROVENANCE_READINESS_CHECKLIST.md`, `OWNER_DECISION_MEMO_F5_H3.md`, `F5_OWNER_DECISION_NOTE.md`, `H3_DIRECTION_OWNER_DECISION_REQUEST.md`, `RUNTIME_LOCK_STATUS.md`, and `TINY_GEMMA_REVALIDATION_AUTH_BLOCKER.md`.

**Decision requested now:** owner selections for **F5** and **H3**, or a documented deferral. **No production pin, approval or scientific authorization is recorded by this packet.**
